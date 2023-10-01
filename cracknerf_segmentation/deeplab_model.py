import torch
import torchvision
from torchvision import models
from torchinfo import summary

#class DeepLabV3Wrapper(torch.nn.Module):
#    def __init__(self, model):
#        super(DeepLabV3Wrapper, self).__init__()
#        self.model = model

#    def forward(self, input):
#        output = self.model(input)['out']
#        return output

def initialize_model(num_classes, keep_feature_extract=False, print_model=False,  use_pretrained=True):
    """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """
    model_deeplabv3 = models.segmentation.deeplabv3_resnet101(weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT, progress=True)
    model_deeplabv3.aux_classifier = None
    if keep_feature_extract:
        for param in model_deeplabv3.parameters():
            param.requires_grad = False

    #input_size = input_size
    model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

    if print_model:
        print("Model's architecture ...")

        print(summary(model_deeplabv3,
             input_size=(32, 3, 256, 256), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
             verbose=0,
             col_names=["input_size", "output_size", "num_params", "trainable"],
             col_width=20,
             row_settings=["var_names"]
                ))



    return model_deeplabv3