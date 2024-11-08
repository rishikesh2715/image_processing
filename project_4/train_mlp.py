import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Define paths to your image folders
numbers_folder = 'templates/numbers'
suits_folder = 'templates/suits'

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        # Construct the full file path
        file_path = os.path.join(folder, filename)
        # Read the image in grayscale mode
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Flatten the image to a 1D array
            img_flat = img.flatten()
            images.append(img_flat)
            # Use the filename (without extension) as the label
            label = os.path.splitext(filename)[0]
            labels.append(label)
    return images, labels

# Load number images and labels
number_images, number_labels = load_images_from_folder(numbers_folder)
# Load suit images and labels
suit_images, suit_labels = load_images_from_folder(suits_folder)

# Prepare data and labels
X_numbers = np.array(number_images)
y_numbers = np.array(number_labels)

X_suits = np.array(suit_images)
y_suits = np.array(suit_labels)

# Encode labels for numbers (e.g., 'A' -> 0, '2' -> 1, ..., 'K' -> 12)
from sklearn.preprocessing import LabelEncoder

number_label_encoder = LabelEncoder()
y_numbers_encoded = number_label_encoder.fit_transform(y_numbers)

suit_label_encoder = LabelEncoder()
y_suits_encoded = suit_label_encoder.fit_transform(y_suits)

# Split data into training and testing sets
X_train_numbers, X_test_numbers, y_train_numbers, y_test_numbers = train_test_split(
    X_numbers, y_numbers_encoded, test_size=0.2, random_state=42)

X_train_suits, X_test_suits, y_train_suits, y_test_suits = train_test_split(
    X_suits, y_suits_encoded, test_size=0.2, random_state=42)

# Define and train the MLP model for numbers
mlp_numbers = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_numbers.fit(X_train_numbers, y_train_numbers)

# Define and train the MLP model for suits
mlp_suits = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_suits.fit(X_train_suits, y_train_suits)

# Evaluate the models
y_pred_numbers = mlp_numbers.predict(X_test_numbers)
accuracy_numbers = accuracy_score(y_test_numbers, y_pred_numbers)
print(f"Accuracy for number classification: {accuracy_numbers * 100:.2f}%")

y_pred_suits = mlp_suits.predict(X_test_suits)
accuracy_suits = accuracy_score(y_test_suits, y_pred_suits)
print(f"Accuracy for suit classification: {accuracy_suits * 100:.2f}%")

# Save the models and label encoders
joblib.dump(mlp_numbers, 'mlp_numbers_model.pkl')
joblib.dump(number_label_encoder, 'number_label_encoder.pkl')

joblib.dump(mlp_suits, 'mlp_suits_model.pkl')
joblib.dump(suit_label_encoder, 'suit_label_encoder.pkl')
