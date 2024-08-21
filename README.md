# Car Driving Simulator - CNN-Based Steering Control

This repository contains the implementation of a Convolutional Neural Network (CNN) model trained to predict discrete steering angles based on processed images from a car driving simulator. The project is organized into several scripts and modules to manage data processing, model training, and interaction with the simulator. The primary goal is to build a model that can be later integrated with a Deep Q-Network (DQN) for reinforcement learning in a discrete action space.


# Introduction
This project leverages a CNN model to predict steering angles in a car driving simulator. The simulator data was transformed and labeled to fit a classification problem with 7 discrete steering options: -60, -30, -15, 0, 15, 30, and 60 degrees. The primary motivation was to later integrate this model with a Deep Q-Network (DQN) for reinforcement learning in a discrete action space.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Data Collection](#data-collection)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Dependencies](#dependencies)
10. [Usage](#usage)
    - [Running the Model](#running-the-model)
    - [Training the Model](#training-the-model)
    - [Image Processing](#image-processing)
11. [Acknowledgements](#acknowledgements)


# Project Structure
The project is organized into the following directory structure:

```
├── avis
│   ├── avisengine.py          # Handles connection to the car simulator
│   ├── config.py              # Configuration settings for the simulator
│   └── utils.py               # Utility functions for the simulator
├── helper
│   └── between_line.py        # Image processing functions for lane detection
├── LICENSE
├── README.md
├── supervised
│   ├── images.zip             # Collected dataset of driving images
│   ├── model.ipynb            # Jupyter notebook for training the CNN model
│   ├── model.py               # CNN model architecture
│   └── sp_model.pth           # Trained model weights
└── Supervised.py              # Main script to start the model and connect to the simulator

```

# Key Files and Directories
* `avis/`: Contains modules for interacting with the car driving simulator.
  * `avisengine.py`: Manages the connection and communication with the simulator.
  * `config.py`: Contains configuration parameters for the simulator.
  * `utils.py`: Utility functions used within the avis module.
* `helper/`: Contains image processing functions.
  * `between_line.py`: Functions for cropping the image, applying thresholds, and resizing the image for lane detection and model input.
* `supervised/`: Contains the data and model-related files.

  * `images.zip`: The dataset of driving images collected from the simulator.
  * `model.ipynb`: A Jupyter notebook used to train the CNN model.
  * `model.py`: Defines the CNN architecture used for classification.
  * `sp_model.pth`: The saved weights of the trained CNN model. 
* `Supervised.py`: The main script that connects the trained model with the simulator.

# Data Collection
The dataset used in this project consists of **5366** images, each labeled with one of the **7 steering angles**. The dataset exhibits class imbalance:

* **0** degrees: **35.1%**
* **15** degrees: **29.3%**
* **-60** degrees: **4.1%**
* **-30** degrees: **6%**
* **-15** degrees: **12.3%**
* **60** degrees: **3.1%**
* **30** degrees: **10%**

# Data Preprocessing
The preprocessing pipeline includes the following steps, implemented in the `helper/between_line.py` script:

1. Image Cropping: Focuses on a specific region of the image to determine if the car is between the lines.

  * `crop_image(x1, x2, y1, y2, image)`: Crops the image based on the given coordinates.
  * `crop_between_line(image)`: Crops a predefined region that contains lane lines.

2. Image Thresholding: Converts the image to the HSV color space and applies thresholding to emphasize lane boundaries.

  * `threshold_using_trackbars(image)`: Uses trackbars to find the best threshold values.
  * `threshold_image(image)`: Applies the optimal threshold values directly.

3. Grayscale Conversion & Resizing: Converts the image to grayscale and resizes it to 50x125 pixels for model input.
  * `resize_image(image)`: Resizes the image to the desired dimensions.
  * `preprocess_image(image)`: Combines resizing and normalization.

4. Image Saving: Saves the processed image to a specified path.
  * `save_image(image, path)`

# Model Architecture
The CNN model is defined in `supervised/model.py` and follows a simple architecture suitable for the small dataset. Below is the architecture summary:
```python
class CNN(nn.Module):
    def __init__(self, n_actions):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(140, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

# Training
The model was trained using the dataset in `supervised/images.zip`. Training was conducted in `supervised/model.ipynb` using the following configuration:

* Loss Function: **Cross-entropy loss**.
* Optimizer: Adam optimizer with a learning rate of **0.001**.
* Epochs: **50**
* Batch Size: **32**

Despite the limited data, the CNN performed well, demonstrating good accuracy on the test set.

# Results
The model achieved high accuracy on the test set, effectively predicting the steering angles based on processed images. The preprocessing steps and simple architecture allowed the model to generalize well despite class imbalances.

# Future Work
Future work includes:

* Integrating the CNN with a Deep Q-Network (DQN) for real-time steering control in the simulator.
* Expanding the dataset with more driving scenarios.
* Experimenting with more complex CNN architectures or other deep learning models to improve performance.
* Addressing class imbalance using techniques such as oversampling or data augmentation.

# Dependencies
* Python 3.8+
* PyTorch 1.9+
* OpenCV 4.5+
* NumPy 1.21+
* opencv-contrib-python
* Pillow
* PySocks
* PyYAML
* regex
* requests

# Usage
### Running the Model
To start the model and connect it to the simulator:

1. Ensure the simulator is running and configured as needed.
2. Run the Supervised.py script:
```python
python Supervised.py
```

# Training the Model
To train the model with your own data:

1. Unzip `images.zip` in the `supervised/` directory.
2. Modify and run the `model.ipynb` notebook to train the model.
3. Save the trained model as `sp_model.pth`.

# Image Processing
The `helper/between_line.py` script contains functions for image processing tasks. You can import and use these functions as needed in your custom scripts.

# Acknowledgements
Special thanks to the developers of PyTorch and OpenCV for providing the tools needed to create this project. Additionally, thanks to Avis Engine team for providing the driving simulator.
