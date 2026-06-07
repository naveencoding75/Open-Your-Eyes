import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ===============================
# STEP 1: Dataset variables setup.
# ===============================

dataset_path = './train'
categories = ['Closed_Eyes', 'Open_Eyes']

# Resizing all images to this size for the convenience ofc.
img_size = (64, 64) 

data = []
labels = []

print("Starting the data pipeline. Hold tight this might take a minute to process 4,000 images...\n")

# ==========================================
# STEP 2: Applying the HOG Extraction Loop.
# ==========================================

# HOG stands for Histogram of Oriented Gradients.

for category_idx, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    
    # Checking if folder exists or not
    if not os.path.exists(folder_path):
        print(f"ERROR: Could not find folder at {folder_path}. Check your path!")
        exit()

    print(f"Processing folder: {category}...")
    
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        
        # 1. Loading the image in Grayscale (B&W).
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # 2. Resizing to 64x64 if not already.
            img_resized = cv2.resize(img, img_size)
            
            # 3. Extracting the HOG Features (The mathematical edges).
            features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
            
            data.append(features)
            # category_idx is 0 for Closed, 1 for Open
            labels.append(category_idx) 

print(f"\nSuccessfully extracted HOG features from {len(data)} total images.")

# ============================
# STEP 3: Training the Model.
# ============================

# Converting lists to NumPy arrays for Scikit-Learn.
X = np.array(data)
y = np.array(labels)

# Spliting the data in 8:2.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Support Vector Machine. Please wait...")
# Used an RBF kernel because facial structures are complex.
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train, y_train)

# ============================
# STEP 4: The Final Auditing.
# ============================
print("\nNow testing the model on unseen data...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\n MODEL ACCURACY: {accuracy * 100:.2f}% ")
print("\nDetailed Diagnostic Report:")
print(classification_report(y_test, predictions, target_names=['Closed', 'Open']))

# Saving the trained model to my pc (hard drive).
joblib.dump(model, 'svm_brain.pkl')
print("\nModel successfully saved as 'svm_brain.pkl'.")