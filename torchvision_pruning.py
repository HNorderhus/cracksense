import os, sys
import torch
import torch.nn as nn
import torch_pruning as tp
import deeplab_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))


if __name__ == "__main__":

    entries = globals().copy()

    def my_prune(model, example_inputs, output_transform, model_name):
        ori_size = tp.utils.count_params(model)
        model.eval()
        ignored_layers = []
        for p in model.parameters():
            p.requires_grad_(True)
        #########################################
        # Ignore unprunable modules
        #########################################
        for m in model.modules():
            if isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == 7:
                ignored_layers.append(m)


        #########################################
        # Build network pruners
        #########################################
        importance = tp.importance.MagnitudeImportance(p=1)
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=1,
            pruning_ratio=0.5,
            global_pruning=False,
            #round_to=round_to,
            #unwrapped_parameters=unwrapped_parameters,
            ignored_layers=ignored_layers,
            #channel_groups=channel_groups,
        )

        #########################################
        # Pruning
        #########################################
        print("==============Before pruning=================")
        #print("Model Name: {}".format(model_name))
        #print(model)

        layer_channel_cfg = {}
        for module in model.modules():
            if module not in pruner.ignored_layers:
                # print(module)
                if isinstance(module, nn.Conv2d):
                    layer_channel_cfg[module] = module.out_channels
                elif isinstance(module, nn.Linear):
                    layer_channel_cfg[module] = module.out_features

        pruner.step()

        print("==============After pruning=================")
        #print(model)

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
                    assert int(0.5 * layer_channel_cfg[module]) == module.out_channels
                elif isinstance(module, nn.Linear):
                    # print(module.out_features, layer_channel_cfg[module])
                    assert int(0.5 * layer_channel_cfg[module]) == module.out_features

            if isinstance(out, (dict, list, tuple)):
                print("  Output:")
                for o in tp.utils.flatten_as_list(out):
                    print(o.shape)
            else:
                print("  Output:", out.shape)
            print("------------------------------------------------------\n")

    successful = []
    unsuccessful = []

    model = deeplab_model.initialize_model(7, keep_feature_extract=True, print_model=False)
    #model.load_state_dict(torch.load("results/models/e1200_baseline.pth"))

    tensor_shape = (32, 3, 224, 224)
    example_inputs = torch.empty(tensor_shape)

    output_transform = lambda x: x["out"]

    try:
        my_prune(
            model, example_inputs=example_inputs, output_transform=output_transform, model_name="deeplabv3"
        )
        successful.append("deeplabv3")
    except Exception as e:
        print(e)
        unsuccessful.append("deeplabv3")
    print("Successful Pruning: %d Models\n" % (len(successful)), successful)
    print("")
    print("Unsuccessful Pruning: %d Models\n" % (len(unsuccessful)), unsuccessful)

    print("Save pruned model")
    torch.save(model, 'results/models/p50_magnitude.pth')

    sys.stdout.flush()

print("Finished!")

print("Successful Pruning: %d Models\n" % (len(successful)), successful)
print("")
print("Unsuccessful Pruning: %d Models\n" % (len(unsuccessful)), unsuccessful)