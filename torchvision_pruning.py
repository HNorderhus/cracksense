import argparse
import os, sys
import torch
import torch.nn as nn
import torch_pruning as tp
import deeplab_model
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
#
#
# if __name__ == "__main__":
#
#     entries = globals().copy()

def my_prune(model, example_inputs, output_transform, model_name, pruning_ratio, p_value, importance_type, iterative_steps):
    ori_size = tp.utils.count_params(model)
    model.eval()
    ignored_layers = []
    for p in model.parameters():
        p.requires_grad_(True)
    #########################################
    # Ignore unprunable modules
    #########################################
    for m in model.modules():
        if isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == 8:
            ignored_layers.append(m)

    #########################################
    # Build network pruners
    #########################################
    # Initialize importance based on the provided importance_type
    # if importance_type == "Taylor":
    #     importance = tp.importance.TaylorImportance()
    # else: # Assuming the other option is Magnitude
    #     importance = tp.importance.MagnitudeImportance(p=p_value)

    if importance_type == "Taylor":
        importance = tp.importance.GroupTaylorImportance()
    elif importance_type == "Magnitude":
        # For Magnitude, you can use GroupNormImportance as it acts as a generalized group-level magnitude importance measure.
        # The 'p' parameter determines the norm degree.
        importance = tp.importance.GroupNormImportance(p=p_value)
    else:
        raise ValueError("Unsupported importance type. Choose 'Taylor' or 'Magnitude'.")

    iterative_steps = iterative_steps

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_ratio,
        global_pruning=False,
        # round_to=round_to,
        # unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
        # channel_groups=channel_groups,
    )

    #########################################
    # Pruning
    #########################################
    print("==============Before pruning=================")
    # print("Model Name: {}".format(model_name))
    # print(model)

    layer_channel_cfg = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:
            # print(module)
            if isinstance(module, nn.Conv2d):
                layer_channel_cfg[module] = module.out_channels
            elif isinstance(module, nn.Linear):
                layer_channel_cfg[module] = module.out_features

    # Taylor Importance requires gradient calculation
    if importance_type == "Taylor":
        # Calculate gradients for importance estimation
        model.zero_grad()
        output = model(example_inputs)
        if output_transform:
            output = output_transform(output)
        loss = output.sum()
        loss.backward()

    for i in tqdm(range(iterative_steps)):

        pruner.step()

    print("==============After pruning=================")
    # print(model)

    #########################################
    # Testing
    #########################################
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

        for module, ch in layer_channel_cfg.items():
            if isinstance(module, nn.Conv2d):
                # print(module.out_channels, layer_channel_cfg[module])
                assert int((1-pruning_ratio) * layer_channel_cfg[module]) == module.out_channels
            elif isinstance(module, nn.Linear):
                # print(module.out_features, layer_channel_cfg[module])
                assert int((1-pruning_ratio) * layer_channel_cfg[module]) == module.out_features

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
    args = parser.parse_args()

    successful = []
    unsuccessful = []

    model = deeplab_model.initialize_model(8, keep_feature_extract=True, print_model=False)
    # model.load_state_dict(torch.load("results/models/e1200_baseline.pth"))

    tensor_shape = (16, 3, 256, 256)
    example_inputs = torch.empty(tensor_shape)

    output_transform = lambda x: x["out"]

    try:
        my_prune(
            model, example_inputs=example_inputs, output_transform=output_transform, model_name=args.model_name, pruning_ratio=args.pruning_ratio,
            p_value = args.p_value, importance_type=args.importance_type, iterative_steps=args.iterative_steps
        )
        successful.append("deeplabv3")
    except Exception as e:
        print(e)
        unsuccessful.append("deeplabv3")
    print("Successful Pruning: %d Models\n" % (len(successful)), successful)
    print("")
    print("Unsuccessful Pruning: %d Models\n" % (len(unsuccessful)), unsuccessful)

    print("Save pruned model")
    torch.save(model, f'results/models/{args.model_name}_baseline.pth')

    sys.stdout.flush()

    print("Finished!")

    print("Successful Pruning: %d Models\n" % (len(successful)), successful)
    print("")
    print("Unsuccessful Pruning: %d Models\n" % (len(unsuccessful)), unsuccessful)


