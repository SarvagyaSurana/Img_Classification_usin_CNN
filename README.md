# Img_Classification_usin_CNN
Convolutional Neural Networks - Image Classification
The objective of this project is to carry out supervised image classification on a collection of colored images. It employs a convolutional neural network design and applies data augmentation and transformations to recognize the category of images from a predefined set of 10 classes.
Data Set (CIFAR-10)

The dataset used is CIFAR-10, which is a widely used benchmark dataset in the field of computer vision and machine learning. It serves as a standard dataset for training and evaluating machine learning algorithms, particularly for image classification tasks.

The dataset has the following features:
    Consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
    Comprises 50,000 training images and 10,000 test images.
    Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
    
STEP 1 - Initialization: importing necessary libraries and modules.

STEP 2 - Loading and Transforming Dataset:
    Loading the dataset from torchvision library using DataLoader:
        Batch size: 32
        Shuffle: True
    Implementing data transformation and augmentation using Compose, as follows:
        Randomly rotating images.
        Randomly flipping images horizontally.
        Randomly changing the brightness, contrast, saturation, and hue of the image (color jitter).
        Scaling the pixel values of the images to be in the range [0, 1].

STEP 3 - Building CNN Model: using nn.Module:
    Input layer.
    Two convolutional layers with ReLU activation function and an increasing number of filters.
    Two max pooling layers following the convolutional layers.
    Flattening layer.
    Two dense/fully connected layers with ReLU activation function.
    Output layer with Softmax activation function.
    Optimizer: Adam.
    Loss function: CrossEntropyLoss.

STEP 4 - Model Training: model is trained using the following configurations:
    Epochs: 25

STEP 5 - Performance Analysis: model accuracy is plotted and analyzed across the epochs.
    Training and validation accuracy across epochs (PyTorch):

