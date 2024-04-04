import argparse
import sys
import traceback
import torch
import torch.nn as nn
import torch_pruning as tp
from tqdm import tqdm
import deeplab_model


def my_prune(model, example_inputs, output_transform, model_name, pruning_ratio, p_value, importance_type,
             iterative_steps):
    """
    Prune the provided model based on the specified parameters, inspired by the structural pruning method introduced by Fang et al. in "Depgraph: Towards any structural pruning" (CVPR 2023).

    Parameters:
        model (torch.nn.Module): The model to be pruned.
        example_inputs (torch.Tensor): Dummy input for the model to infer the computational graph.
        output_transform (callable): Function to transform the output for loss calculation, if necessary.
        model_name (str): Name of the model for logging.
        pruning_ratio (float): The ratio of parameters to prune.
        p_value (int): The 'p' value for MagnitudeImportance, if used.
        importance_type (str): Type of importance ('Taylor' or 'Magnitude') to be used for pruning.
        iterative_steps (int): Number of iterative steps to prune.

    Returns:
        None

    Reference:
        Fang, Gongfan et al. "Depgraph: Towards any structural pruning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    """
    ori_size = tp.utils.count_params(model)
    model.eval()

    # Set gradients for all parameters to True and collect layers to ignore
    ignored_layers = []
    for p in model.parameters():
        p.requires_grad_(True)

    #exclude final layer from the pruning
    for m in model.modules():
        if isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == 8:
            ignored_layers.append(m)

    # Select importance criterion
    if importance_type == "Taylor":
        importance = tp.importance.TaylorImportance()
    elif importance_type == "Magnitude":
        importance = tp.importance.MagnitudeImportance(p=p_value)
    else:
        raise ValueError("Unsupported importance type. Choose 'Taylor' or 'Magnitude'.")

    # Initialize pruner
    pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            pruning_ratio=pruning_ratio,
            global_pruning=False,
            ignored_layers=ignored_layers,)

    # Pre-pruning status
    print("==============Before pruning=================")

    layer_channel_cfg = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:
            # print(module)
            if isinstance(module, nn.Conv2d):
                layer_channel_cfg[module] = module.out_channels
            elif isinstance(module, nn.Linear):
                layer_channel_cfg[module] = module.out_features

    # If Taylor importance, need to calculate gradients with respect to output
    if importance_type == "Taylor":
        model.zero_grad()
        output = model(example_inputs)
        if output_transform:
            output = output_transform(output)
        loss = output.sum()
        loss.backward()

    for _ in tqdm(range(iterative_steps)):
        pruner.step()

    # Post-pruning status
    print("==============After pruning=================")
    with torch.no_grad():
        if isinstance(example_inputs, dict):
            out = model(**example_inputs)
        else:
            out = model(example_inputs)
        if output_transform:
            out = output_transform(out)
        print("{} Pruning: ".format(model_name))
        params_after_prune = tp.utils.count_params(model)
        print("  Params: %s => %s" % (ori_size, params_after_prune))

        if isinstance(out, (dict, list, tuple)):
            print("  Output:")
            for o in tp.utils.flatten_as_list(out):
                print(o.shape)
        else:
            print("  Output:", out.shape)
        print("------------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruning_ratio", type=float, help="Pruning ratio for the model")
    parser.add_argument("--importance_type", type=str, help="Taylor or Magnitude")
    parser.add_argument("--p_value", type=int, help="p value for MagnitudeImportance")
    parser.add_argument("--model_name", type=str, help="Name for saving the pruned model")
    parser.add_argument("--iterative_steps", type=int, default=1, help="number of iterations")
    parser.add_argument("--state_dict", type=str, help="Path to the state_dict file")
    parser.add_argument("--keep_feature_extract", type=bool, default=True, help="Keep feature extraction layers frozen")

    args = parser.parse_args()

    successful = []
    unsuccessful = []

    model = deeplab_model.initialize_model(8, keep_feature_extract=args.keep_feature_extract)
    if args.state_dict:
        model.load_state_dict(torch.load(args.state_dict))

    tensor_shape = (8, 3, 512, 512)
    example_inputs = torch.empty(tensor_shape)

    output_transform = lambda x: x["out"]

    try:
        my_prune(
            model, example_inputs=example_inputs, output_transform=output_transform, model_name=args.model_name,
            pruning_ratio=args.pruning_ratio,
            p_value=args.p_value, importance_type=args.importance_type, iterative_steps=args.iterative_steps)
        successful.append("deeplabv3")
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc()
        unsuccessful.append("deeplabv3")

    print("Successful Pruning: %d Models\n" % (len(successful)), successful)
    print("")
    print("Unsuccessful Pruning: %d Models\n" % (len(unsuccessful)), unsuccessful)
    print("Save pruned model")
    torch.save(model, f'results/models/{args.model_name}_model.pth')
    sys.stdout.flush()
    print("Finished!")