import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
from exception import ImageProcessingError, ModelPredictionError, FileOpenError
from logging_setup import setup_logging

# Setup logging
setup_logging()

# Load the model
model_path = 'model/age_gender_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise ModelPredictionError(f"Error loading model: {e}")

# Function to preprocess the image
def preprocess_image(img_path):
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ImageProcessingError(f"Error reading image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (48, 48))
        image = image / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise ImageProcessingError(f"Error processing image: {e}")

# Function to predict gender and age
def predict_gender_age(img_path):
    try:
        image = preprocess_image(img_path)
        gender, age = model.predict(image)
        gender = 'Male' if gender[0] > 0.5 else 'Female'
        age = int(age[0])
        return gender, age
    except Exception as e:
        logging.error(f"Error predicting gender and age: {e}")
        raise ModelPredictionError(f"Error predicting gender and age: {e}")

# Function to open an image file
def open_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            raise FileOpenError("No file selected.")
        img = Image.open(file_path)
        img = img.resize((200, 200), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img
        gender, age = predict_gender_age(file_path)
        result_text.set(f"Gender: {gender}, Age: {age}")
    except FileOpenError as e:
        logging.warning(e)
        messagebox.showwarning("File Open Error", str(e))
    except (ImageProcessingError, ModelPredictionError) as e:
        logging.error(e)
        messagebox.showerror("Error", str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        messagebox.showerror("Error", f"Unexpected error: {e}")

# Create the main window
root = tk.Tk()
root.title("Age and Gender Prediction")
root.geometry("500x400")  # Increase the initial window size

# Add a label to display the image
panel = tk.Label(root)
panel.pack()

# Add a button to open an image file
btn = tk.Button(root, text="Open Image", command=open_image)
btn.pack()

# Add a label to display the prediction result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 16))
result_label.pack()

# Run the GUI main loop
root.mainloop()
