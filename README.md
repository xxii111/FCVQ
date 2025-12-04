# Transform-Free-Feature-Coding-via-Entropy-Constrained-Vector-Quantization

This project is the official implementation of the paper titled “Transform-Free Feature Coding via Entropy-Constrained Vector Quantization”.

# Feature Dataset
For the ResNet50-Cls feature dataset, we used the ResNet50 pretrained weights (V1) provided by PyTorch to extract features — you can download them from the official [PyTorch](https://docs.pytorch.org/vision/main/models.html). We used the output of the last BatchNorm layer in layer4 as the feature representation, with a shape of **2048 × 7 × 7**.
Both the training and testing feature datasets were extracted from [ImageNet](https://www.image-net.org/).  

For the DINOv2-Cls and DINOv2-Seg feature dataset, we followed the setup in the [LaMoFC](https://github.com/chansongoal/LaMoFC/) for using DINOv2 as a visual feature extractor.

# Environment Set Up
To ensure training and testing, please make sure your environment supports both[CompressAI](https://github.com/InterDigitalInc/CompressAI) and [DINOv2](https://github.com/facebookresearch/dinov2).

# Usage
You can perform both training and testing by running the `.sh` scripts, and you may need to modify the corresponding input arguments as required.

# Acknowledgement
Special thanks to Runsen Feng.