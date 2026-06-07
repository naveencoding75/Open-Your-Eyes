import cv2
import joblib
import numpy as np
import gradio as gr
from skimage.feature import hog

# 1. Wake up the "Brain"
# This loads the model you trained earlier
model = joblib.load('svm_brain.pkl')

# 2. The Core Prediction Function
def analyze_eye(input_image):
    # SAFETY NET: If no image is uploaded, stop here and tell the user.
    if input_image is None:
        return "⚠️ PLEASE UPLOAD AN IMAGE FIRST"
        
    # Gradio sends the image as an RGB numpy array. 
    # We must convert it to grayscale to match how we trained the model.
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    
    # Resize to the exact dimensions the SVM expects
    resized_img = cv2.resize(gray_img, (64, 64))
    
    # Extract the mathematical HOG features
    features = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    # The SVM needs a 2D array, so we reshape it
    features_reshaped = features.reshape(1, -1)
    
    # Ask the SVM for its diagnosis
    prediction = model.predict(features_reshaped)[0]
    
    # Format the output for the UI
    if prediction == 0:  # Remember: 0 was Closed, 1 was Open in our training loop
        return "🔴 ALERT: DROWSY (EYES CLOSED)"
    else:
        return "🟢 DRIVER AWAKE (EYES OPEN)"
    
# 3. Build the Web Dashboard
# This creates the sleek Next.js-style UI automatically
dashboard = gr.Interface(
    fn=analyze_eye, 
    inputs=gr.Image(label="Upload Driver Camera Feed (Eye Image)"), 
    outputs=gr.Textbox(label="SVM System Diagnosis", text_align="center"),
    title="SVM Drowsiness Detection System",
    description="Upload an image of an eye to test the Support Vector Machine's biometric classification.",
    theme="default" # Gives it a clean, modern look
)

# 4. Launch the App!
if __name__ == "__main__":
    print("Starting the web server...")
    dashboard.launch()