import torchvision
from torchvision import models


def initialize_model(num_classes, keep_feature_extract=True):
    """
    Initialize a DeepLabV3 model with a ResNet101 backbone for semantic segmentation.
    Parameters:
    num_classes: int - The number of classes for the segmentation task.
    keep_feature_extract: bool - If True, the model is set to feature extraction mode, freezing all model parameters.
    """
    model_deeplabv3 = models.segmentation.deeplabv3_resnet101(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT, progress=True)

    # If the model is set for feature extraction, freeze all model parameters to prevent updating during training.
    if keep_feature_extract:
        for param in model_deeplabv3.parameters():
            param.requires_grad = False

    model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
    model_deeplabv3.aux_classifier = None
    return model_deeplabv3
