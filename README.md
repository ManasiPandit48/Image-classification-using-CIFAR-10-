# Image-Classification-Training

This project involves creating and training a Convolutional Neural Network (CNN) to classify images from a given dataset. The trained model can then be used to perform real-time image classification.

## Requirements

The following packages are required to run this project:

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages using pip:

```bash
pip install numpy keras tensorflow matplotlib scikit-learn
```

## Dataset

We have used CIFAR-10 dataset which is provided by Tensorflow library.

## Training the Image Classification Model

The provided Jupyter Notebook is used to train the image classification model. It reads the dataset, builds a Convolutional Neural Network (CNN) model, trains the model on the training data, and evaluates its performance.

### Training Steps:

1. **Import Libraries**: Import all the necessary libraries and modules required for the project.
2. **Load and Preprocess Data**: Load the dataset and perform preprocessing steps such as normalization and data augmentation.
3. **Build the CNN Model**: Define a sequential model with convolutional layers, max-pooling layers, dropout layers, and dense layers.
4. **Compile the Model**: Configure the model with a loss function, optimizer, and metrics.
5. **Train the Model**: Fit the model using the training data and validate using the validation data.
6. **Evaluate the Model**: Evaluate the model's performance on the test data.
7. **Save the Trained Model**: Save the model architecture and weights.

### Running the Training Notebook:

To train the model, follow the steps provided in the notebook. Make sure to run each cell sequentially. The training process might take some time depending on your hardware.

## Acknowledgments

This project uses TensorFlow and Keras for building and training the neural network. The CIFAR-10 dataset used in this project is publicly available and commonly used for educational purposes.

---
