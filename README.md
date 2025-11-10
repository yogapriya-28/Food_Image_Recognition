# Food_Image_Recognition

# 🍔 Food Image Recognition using InceptionV3 (Food-101 Dataset)

## 📘 Overview
This project implements an **AI-powered food image recognition system** that classifies images into **101 different food categories** using **Transfer Learning** with **InceptionV3**.  
The model is trained on the **Food-101 dataset**, which contains over 100,000 labeled food images.

---
## Project Description
This project implements a food image recognition system that classifies images into 101 food categories using the Food-101 dataset. The model is based on the InceptionV3 architecture and is trained to optimize classification accuracy. Additional functionalities include visualization of convolutional layer activations and class activation maps to understand model decisions.

## Dataset
The Food-101 dataset contains 101,000 images of 101 food categories. It is publicly available at:
[Food-101 dataset](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)

## Installation and Dependencies
This project requires Python 3.x and the following libraries:
- tensorflow
- keras
- opencv-python
- scikit-image
- numpy
- matplotlib
- seaborn
- sklearn
- ipywidgets
- h5py

Install dependencies using:

## Setup and Usage

1. Download and extract the Food-101 dataset:

2. Generate train and test splits using provided methods in the notebook.

3. Load training and testing images with resizing to 200x200 pixels.

4. Train the model using the InceptionV3 architecture with the specified parameters:
   - Batch size: 16
   - Epochs: 30
   - Optimizer: SGD with learning rate 0.0001 and momentum 0.9

5. Save the best model checkpoints during training.

6. Use the trained model to make predictions on new test images.

7. Visualize activations and heatmaps for interpretability.

## Core Functions and Methods
- `gen_dir_file_map(path)`: Generates a directory-to-file map from text files listing image paths.
- `copytree(source, target, symlinks=False, ignore=None)`: Recursively copies directories, ignoring specified files.
- `load_images(path_to_imgs)`: Loads and resizes images into an array along with their class indices.
- `gen_train_test_split(path_to_imgs, target_path)`: Creates train/test folders from dataset.
- `load_train_test_data(path_to_train_imgs, path_to_test_imgs)`: Loads train and test datasets.
- `predict_class(model, images, show=True)`: Predicts classes for given images and optionally displays them.
- Visualization utilities to generate activation layer outputs, heatmaps, and class activation maps.

## Model Architecture Summary
- Based on pre-trained InceptionV3 (ImageNet weights, excluding top).
- Added GlobalAveragePooling, Dense with 128 units, Dropout, and final Dense layer with 101 softmax outputs.
- Regularization applied on output layer with L2 penalty.

## Example Predictions
The notebook downloads sample images such as cupcakes, pizza, french fries, garlic bread, strawberry shortcake, and spaghetti carbonara to demonstrate prediction performance.

## Visualization
- Activation layers and filters can be visualized to understand what the network learned.
- Heatmaps and class activation maps highlight important regions of input images influencing model predictions.

## File Structure
- `food-101/` - Dataset directory containing images and meta files.
- `best_model_101class.hdf5` - Best model weights saved during training.
- `history_101class.log` - Training logs.
- Notebook and scripts for data processing, training, prediction, and visualization.

## References
- Food-101 Dataset: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
- InceptionV3 Paper: https://arxiv.org/abs/1512.00567
---






