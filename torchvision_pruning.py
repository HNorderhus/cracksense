import argparse
import sys
import traceback

import torch
import torch.nn as nn
import torch_pruning as tp
from tqdm import tqdm

import deeplab_model


def my_prune(model, example_inputs, output_transform, model_name, pruning_ratio, p_value, importance_type,
             iterative_steps, prune_layers, pruning_ratio_layer1, pruning_ratio_layer2):
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

    if importance_type == "Taylor":
        importance = tp.importance.TaylorImportance()
    elif importance_type == "Magnitude":
        # For Magnitude, you can use GroupNormImportance as it acts as a generalized group-level magnitude importance measure.
        # The 'p' parameter determines the norm degree.
        importance = tp.importance.MagnitudeImportance(p=p_value)
    else:
        raise ValueError("Unsupported importance type. Choose 'Taylor' or 'Magnitude'.")

    iterative_steps = iterative_steps

    if prune_layers:
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            pruning_ratio=pruning_ratio,
            pruning_ratio_dict={model.backbone.layer1: pruning_ratio_layer1, model.backbone.layer2: pruning_ratio_layer2},
            global_pruning=False,
            ignored_layers=ignored_layers,
            # channel_groups=channel_groups,
        )
    else:
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            pruning_ratio=pruning_ratio,
            global_pruning=False,
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
    parser.add_argument("--prune_layers", type=bool, default=False, help="Enable specific pruning")
    parser.add_argument("--pruning_ratio_layer1", type=float, help="Specific pruning ratios for layer 1")
    parser.add_argument("--pruning_ratio_layer2", type=float, help="Specific pruning ratios for layer 2")

    args = parser.parse_args()

    successful = []
    unsuccessful = []

    model = deeplab_model.initialize_model(8, keep_feature_extract=True, print_model=False)
    # model.load_state_dict(torch.load("results/models/e1200_baseline.pth"))

    tensor_shape = (16, 3, 512, 512)
    example_inputs = torch.empty(tensor_shape)

    output_transform = lambda x: x["out"]

    try:
        my_prune(
            model, example_inputs=example_inputs, output_transform=output_transform, model_name=args.model_name,
            pruning_ratio=args.pruning_ratio,
            p_value=args.p_value, importance_type=args.importance_type, iterative_steps=args.iterative_steps,
            prune_layers = args.prune_layers, pruning_ratio_layer1 = args.pruning_ratio_layer1,
            pruning_ratio_layer2 = args.pruning_ratio_layer2
        )
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
    torch.save(model, f'results/models/{args.model_name}_baseline.pth')

    sys.stdout.flush()

    print("Finished!")