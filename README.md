# Project_4
# Using NN to Detect Pneumonia in Chest X-rays 

## Description :

“Convolutional neural networks (CNNs) are effective tools for image understanding.
They have outperformed human experts in many image understanding tasks” 
(Sarvamangala & Kulkarni, 2021).

The goal of the project is use Deep Learning to built a Neural network to detect
abnormalities is chest X-rays.
The ways we can detect if a chest X-ray has Pneumonia is either by having a doctor
look at the chest X_ray exam or have a radiologist interpret the result of an X-ray.
Those two ways relay on human experts in image understanding.
In this project we are going to utilize NN as an effective tool for image Understanding.

## Exploratory Data Analysis:

The data we used is chest X-rays images. Those images contains 2 classes: 
1- normal chest X-rays 
2- Pneumonia chest X-rays 
Checked the data for Class balance:
Training Data : 
Total images = 5,219 / 25.7% Normal / 74.2% Pneumonia 
Test Data : 
Total images = 627   / 37.4% Normal / 62.3% Pneumonia 

## Modeling :

### Base Model :

Started the Base Model with a simple model structure. It consisted of 2 Dense Layers,
an Input Layer and an Output Layer. For the Parameters we utilized Talos Library which helped by fully automating hyperparameter tuning.
with the help of Talos library we were able to use 3 different inputs for the Nodes('nodes_dense': [12, 20, 100]), 2 different inputs for the Activation function ('activation_dense': ['relu', 'elu']), and 2 different inputs for the optimizer ('optimizer': ['adam', 'sgd']). 

Talos ran 216 models, which we were able to sort by the accuracy of the model.
After finding the model with the best accuracy we were able to see the best parameters to help give us our best model. 
The best model Results:  Training Accuracy = 0.953164 / Validation Accuracy = 0.962300
Grest result no sign of over or under fitting. 

### Model_2 :

We decided to make the model structure more complex with the hope of improving the model Accuracy, by adding 1 CNN layer and a pooling layer, and 1 Drop out layer.
kept using Talos for automating hyoerparameter tuning. For the CNN Nodes used 
('nodes_cnn':[32, 64,128]), for CNN activation functions used ('activation_cnn': ['relu', 'softmax']), and for the Drop out used (dropout': [0.1, 0.25, 0.5]).

Talos ran another 216 models, sort them by the Accuracy or the model and we got,
Best model results: Training Accuracy = 0.999726 / Validation Accuracy = 0.971246
As you can see model Accuracy did import by 1%, but the training accuracy is higher then and validation accuracy which is a sign of over fitting. 

### Model_3 :






