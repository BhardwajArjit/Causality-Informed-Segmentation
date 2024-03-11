# Semantic Segmentation with ViT on Aerial Imagery Dataset

This repository contains the implementation of semantic segmentation on the aerial imagery dataset obtained from MBRSC satellites over Dubai using the Vision Transformer (ViT) architecture. Semantic segmentation is a computer vision task that involves classifying each pixel in an image into a specific category or class. The dataset comprises 72 high-resolution images annotated with pixel-wise semantic segmentation into 6 classes: Building, Land, Road, Vegetation, Water, and Unlabeled.

## Dataset:

The aerial imagery dataset is publicly available on Kaggle and can be accessed through the [dataset link](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery). The images were obtained from MBRSC satellites over Dubai and are accompanied by pixel-wise annotations for various land cover classes. This dataset provides valuable insights for urban planning, environmental monitoring, and land use analysis in urban areas.

**Downloading Instructions:**

To download the dataset, you need to follow these steps:
1. Create a Kaggle account if you don't have one.
2. Visit the [Aerial Imagery Dataset page](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) on Kaggle.
3. Enter your Kaggle username and API key when prompted in Google Colab.

## Architecture:

The Vision Transformer (ViT) architecture is employed for semantic segmentation. ViT is a transformer-based model that has shown promising results in various computer vision tasks, including image classification and object detection. By leveraging self-attention mechanisms, ViT can effectively capture long-range dependencies in the aerial imagery, facilitating accurate segmentation of land cover classes.

## Tools and Libraries:

- **Google Colab**: The implementation is carried out entirely in Google Colab, leveraging its GPU runtime for accelerated training.
- **PyTorch**: PyTorch is used as the deep learning framework for building and training the ViT model.
- **torchvision**: The torchvision library provides access to popular datasets, pretrained models, and image transformation utilities, facilitating dataset loading and preprocessing.
- **NumPy**: NumPy is used for numerical computations and array operations.

## Usage:

To replicate the experiment:

1. Clone this repository to your Google Drive.
2. Open and run the notebooks in Google Colab, following the instructions provided within each notebook.

## Results:

Upon training the ViT model on the aerial imagery dataset, the performance can be evaluated using various metrics such as mean Intersection over Union (mIoU). Additionally, qualitative evaluation through visual inspection of segmentation results provides insights into the model's ability to accurately segment land cover classes.

## Contributors:

- [Arjit Bhardwaj](https://github.com/BhardwajArjit)
