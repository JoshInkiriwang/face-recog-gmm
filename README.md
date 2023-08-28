# Facial Recognition with Gaussian Mixture Model (GMM) and Backpropagation

This repository contains the implementation of a facial recognition system using Gaussian Mixture Model (GMM) and Backpropagation. The goal of this project is to recognize faces from input images and achieve a high accuracy rate. The initial implementation uses GMM for feature extraction and Backpropagation for classification. However, due to issues with low accuracy, additional approaches such as Convolutional Neural Networks (CNNs) are recommended for improving the accuracy.

## Project Overview

The project consists of the following components:

1. **Notebook File**: The main notebook file is `Final-Project_GMM-Backpropagation.ipynb`, which contains the implementation of the facial recognition system using GMM and Backpropagation.

2. **Data**: The project uses facial images for training and testing. The data should be stored in a directory named `data/` within the repository. Due to the observed low accuracy, it is recommended to use images with higher resolutions for better performance.

3. **Results**: The results of the facial recognition system, including accuracy scores and any visualization, will be saved within the notebook and can be viewed during or after the execution.

## Getting Started

To run the notebook and experiment with the facial recognition system:

1. Clone the repository using the following command:
   ```
   git clone https://github.com/JoshInkiriwang/face-recog-gmm.git
   ```

2. Make sure you have all the required dependencies installed. You can install them using the following command:
   ```
   pip install -r requirements.txt
   ```

3. Place your facial image data in the `data/` directory. Ensure that you use images with higher resolutions to potentially improve accuracy.

4. Open the `Final-Project_GMM-Backpropagation.ipynb` notebook in a Jupyter environment.

5. Execute the notebook cells to run the facial recognition system using GMM and Backpropagation.

## Improving Accuracy

As observed, the current implementation's accuracy is below 50% due to the limitations of using GMM and Backpropagation for facial recognition. To enhance accuracy, consider implementing the following approaches:

1. **Convolutional Neural Networks (CNNs)**: Replace the GMM-Backpropagation pipeline with a CNN-based approach. CNNs are well-suited for image recognition tasks and can capture complex features present in faces.

2. **Data Augmentation**: Apply data augmentation techniques to artificially increase the diversity of the training dataset. This can help the model generalize better to new faces.

3. **Pre-trained Models**: Utilize pre-trained CNN models (e.g., VGG, ResNet) trained on large face datasets. Fine-tune these models on your dataset to leverage their learned features.

4. **Hyperparameter Tuning**: Experiment with different hyperparameters for your models, such as learning rates, batch sizes, and optimization algorithms.

5. **Ensemble Methods**: Combine predictions from multiple models or approaches using ensemble techniques to improve overall accuracy.

## Repository Structure

```
/
├── data/                   # Directory for storing facial image data
├── final_project.ipynb     # Main notebook for the project
├── requirements.txt        # List of dependencies
└── README.md               # This README file
```

## Acknowledgments

This project was developed by Joshua Verbiano Inkiriwang as a final project for Multimedia Nusantara University. The implementation is a starting point, and future improvements can lead to better accuracy in facial recognition.

If you have any questions or suggestions, feel free to inkiriwangjosh21@gmail.com

Repository URL: [https://github.com/JoshInkiriwang/face-recog-gmm](https://github.com/JoshInkiriwang/face-recog-gmm)
