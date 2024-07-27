# Gender and Age Detection

This project implements a GUI application for detecting gender and age from an image using a trained deep learning model. The model is built using TensorFlow/Keras and can predict the gender and age of a person in a given image.

## Features

- Load and preprocess images.
- Predict gender and age using a pre-trained model.
- Display the predicted gender and age in the GUI.
- Drag and drop images into the GUI for prediction.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- OpenCV
- Pillow
- Tkinter

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/prabhakarsharma-pythonaire/Gender-and-age-detection.git
    cd Gender-and-age-detection
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the trained model saved at `model/age_gender_model.h5`.

2. Run the application:
    ```sh
    python main.py
    ```

3. Use the GUI to open an image file or drag and drop an image. The predicted gender and age will be displayed.

## Project Structure

