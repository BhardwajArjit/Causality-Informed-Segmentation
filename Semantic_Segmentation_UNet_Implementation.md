# Semantic Segmentation with UNet on Cityscapes Dataset

This repository contains the implementation of semantic segmentation on the Cityscapes dataset using the UNet architecture in Google Colab. Semantic segmentation is a computer vision task that involves classifying each pixel in an image into a specific category or class. The Cityscapes dataset consists of urban street scenes, with pixel-level annotations for various classes such as road, sidewalk, car, and pedestrian.

## Dataset:

The Cityscapes dataset is publicly available and can be accessed through the [official website](https://www.cityscapes-dataset.com/). However, for convenience, this implementation utilizes the Cityscapes dataset hosted on Kaggle. The dataset comprises high-resolution images of urban scenes, along with pixel-wise annotations for 30 different classes.

**Downloading Instructions:**

To download the dataset, you need to follow these steps:
1. Create a Kaggle account if you don't have one.
2. Visit the [Cityscapes Dataset page](https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs) on Kaggle.
3. Enter your Kaggle username and API key when prompted in Google Colab.

## Architecture:

The UNet architecture is employed for semantic segmentation. UNet is a fully convolutional neural network (FCN) architecture that is popular for semantic segmentation tasks due to its ability to capture both local and global features effectively. It consists of a contracting path to capture context and a symmetric expanding path for precise localization.

## Tools and Libraries:

- **Google Colab**: The implementation is carried out entirely in Google Colab, leveraging its GPU runtime for accelerated training.
- **PyTorch**: PyTorch is used as the deep learning framework for building and training the UNet model.
- **torchvision**: The torchvision library provides access to popular datasets, pretrained models, and image transformation utilities, facilitating dataset loading and preprocessing.
- **NumPy**: NumPy is used for numerical computations and array operations.


## Usage:

To replicate the experiment:

1. Clone this repository to your Google Drive.
2. Open and run the notebooks in Google Colab, following the instructions provided within each notebook.

## Results:

Upon training the UNet model on the Cityscapes dataset, the performance can be evaluated using various metrics such as mean Intersection over Union (mIoU).

## Contributors:

- [Arjit Bhardwaj](https://github.com/BhardwajArjit)
