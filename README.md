# AI Handwritten Character Recognition (Tensorflow)

This repository contains the Python code for a simple AI model that can recognize handwritten digits (0-9). The model is built using the TensorFlow and Keras libraries.

**This was a guide project.**

## Overview

This project demonstrates a basic implementation of a feedforward neural network to classify handwritten digits from the MNIST dataset. It includes:

* A Python script `main.py` that loads the MNIST dataset, preprocesses the images, defines and trains a neural network model, evaluates its performance, saves the trained model, and makes predictions on sample images.

## Getting Started

To get started with this project, you will need to have Python and the necessary libraries installed on your system.

### Prerequisites

* **Python 3.x:** Download and install the latest version of Python from [https://www.python.org/downloads/](https://www.python.org/downloads/)
* **TensorFlow:** Install TensorFlow using pip:
    ```bash
    pip install tensorflow
    ```
* **NumPy:** Install NumPy using pip:
    ```bash
    pip install numpy
    ```
* **OpenCV (cv2):** Install OpenCV using pip:
    ```bash
    pip install opencv-python
    ```
* **Matplotlib:** Install Matplotlib using pip:
    ```bash
    pip install matplotlib
    ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <ttps://github.com/Ra-Verse/AI-Handwritten-Character-Recognition/tree/main>
    cd AI-Handwritten-Character-Recognition-Tensorflow
    ```
    
2.  **(Optional) Create a virtual environment:** It is recommended to create a virtual environment to isolate the project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy opencv-python matplotlib
    ```

4.  **Save the provided Python code as `main.py` in the project directory.**

5.  **Prepare sample images:** To test the prediction functionality, you will need to create a few PNG image files named `1.png`, `2.png`, `3.png`, `4.png`, and `5.png` in the same directory as `main.py`. These images should contain single handwritten digits on a white background (or a background that will be inverted to white).

## Usage

1.  **Run the `main.py` script:**
    ```bash
    python main.py
    ```

    This script will:
    * Load and preprocess the MNIST dataset.
    * Define and train a simple neural network model.
    * Evaluate the model's accuracy and loss on the test dataset.
    * Save the trained model as `digits.model.keras`.
    * Load the sample images (`1.png` to `5.png`), preprocess them, make predictions using the trained model, and display the predicted digit along with the image.

## Model Architecture

The model is a sequential neural network with the following layers:

1.  **Flatten:** Converts the 28x28 pixel input images into a 784-dimensional vector.
2.  **Dense (128 units, ReLU activation):** A fully connected layer with 128 neurons and the Rectified Linear Unit (ReLU) activation function.
3.  **Dense (128 units, ReLU activation):** Another fully connected layer with 128 neurons and ReLU activation.
4.  **Dense (10 units, Softmax activation):** The output layer with 10 neurons (one for each digit 0-9) and the Softmax activation function, which outputs the probability distribution over the classes.

The model is compiled using the Adam optimizer, sparse categorical cross-entropy loss (suitable for integer labels), and accuracy as the evaluation metric.

## Dataset

This project utilizes the **MNIST** dataset, which consists of 70,000 grayscale images of handwritten digits (0-9). The dataset is automatically downloaded and loaded using `tf.keras.datasets.mnist`.

## Saving the Model

The trained model is saved to a file named `digits.model.keras` using `model.save('digits.model.keras')`. This allows you to load and reuse the trained model without retraining it.

## Making Predictions on Custom Images

The script includes a loop that reads images named `1.png` through `5.png`, preprocesses them (inverting the colors and normalizing the pixel values), and uses the trained model to predict the digit in each image. The predicted digit and the corresponding image are then displayed using Matplotlib.

**Note:** Ensure that the sample PNG images have a format that OpenCV can read and contain a single, clear handwritten digit. The preprocessing steps in the code assume the digit is initially dark on a light background.

## Contributing

Contributions to this project are welcome! If you find any bugs or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

MIT License, Apache License 2.0

## Acknowledgements

This project was a guided project by NuralNine

* TensorFlow library: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* Keras API: [https://keras.io/](https://keras.io/)
* MNIST dataset: Yann LeCun and Corinna Cortes, "MNIST handwritten digit database," AT&T Labs [Online]. Available: http://yann.lecun.com/exdb/mnist/
* NumPy library: [https://numpy.org/](https://numpy.org/)
* OpenCV library: [https://opencv.org/](https://opencv.org/)
* Matplotlib library: [https://matplotlib.org/](https://matplotlib.org/)
