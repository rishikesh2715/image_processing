# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import cv2
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# import pickle

# class CardDataProcessor:
#     def __init__(self, image_size=(50, 50), threshold=128):
#         self.image_size = image_size
#         self.threshold = threshold
#         self.suit_encoder = LabelEncoder()
        
#     def preprocess_image(self, image_path):
#         """Load, resize and binarize image"""
#         img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, self.image_size)
#         return img.flatten() / 255.0  # Flatten and normalize
    
#     def load_data(self, data_dir):
#         """Load suit images"""
#         images = []
#         labels = []
#         data_path = Path(data_dir)
        
#         print("\nLoading suit images...")
#         for class_dir in data_path.iterdir():
#             if class_dir.is_dir():
#                 print(f"Processing class: {class_dir.name}")
#                 count = 0
#                 for img_path in class_dir.glob('*.*'):  # This will get both .jpg and .png
#                     images.append(self.preprocess_image(img_path))
#                     labels.append(class_dir.name)
#                     count += 1
#                 print(f"  Found {count} images")
        
#         return np.array(images), labels
    
#     def prepare_data(self, data_dir):
#         """Prepare suit dataset"""
#         # Load suits
#         X_suits, y_suits = self.load_data(data_dir)
#         y_suits_encoded = self.suit_encoder.fit_transform(y_suits)
        
#         # Split data
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_suits, y_suits_encoded, test_size=0.2, random_state=42)
        
#         return X_train, X_val, y_train, y_val
    
#     def visualize_samples(self, X, y, title):
#         """Visualize sample images from dataset"""
#         plt.figure(figsize=(15, 5))
#         for i in range(5):
#             plt.subplot(1, 5, i+1)
#             img = X[i].reshape(self.image_size)
#             plt.imshow(img, cmap='gray')
#             plt.title(f'Class: {self.suit_encoder.inverse_transform([y[i]])[0]}')
#             plt.axis('off')
#         plt.suptitle(title)
#         plt.show()

# def create_mlp_model(input_shape, num_classes):
#     """Create MLP model with specified architecture"""
#     model = models.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(256, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(num_classes, activation='softmax')
#     ])
    
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     return model

# def plot_training_history(history, title):
#     """Plot training history"""
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title(f'{title} - Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title(f'{title} - Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()

# def plot_confusion_matrix(y_true, y_pred, encoder, title):
#     """Plot confusion matrix"""
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', 
#                 xticklabels=encoder.classes_,
#                 yticklabels=encoder.classes_)
#     plt.title(f'{title} Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.show()

# def main():
#     # Configuration
#     IMAGE_SIZE = (50, 50)
#     THRESHOLD = 128
#     EPOCHS = 50
    
#     # Get data directory
#     data_dir = input("Enter the path to training data directory: ")
#     save_dir = input("Enter the path to save model and encoder: ")
#     Path(save_dir).mkdir(parents=True, exist_ok=True)
    
#     # Initialize processor and load data
#     processor = CardDataProcessor(IMAGE_SIZE, THRESHOLD)
#     X_train, X_val, y_train, y_val = processor.prepare_data(data_dir)
    
#     # Visualize sample images
#     processor.visualize_samples(X_train, y_train, "Sample Suit Images")
    
#     # Train suit classifier
#     print("\nTraining suit classifier...")
#     model = create_mlp_model(IMAGE_SIZE[0] * IMAGE_SIZE[1], 
#                            len(processor.suit_encoder.classes_))
    
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=EPOCHS,
#         callbacks=[
#             EarlyStopping(patience=10, restore_best_weights=True),
#             ModelCheckpoint(f"{save_dir}/suit_model_best.h5", 
#                           save_best_only=True)
#         ]
#     )
    
#     # Plot training history
#     plot_training_history(history, "Suit Classifier")
    
#     # Generate predictions and plot confusion matrix
#     y_pred = model.predict(X_val).argmax(axis=1)
#     plot_confusion_matrix(y_val, y_pred, processor.suit_encoder, "Suit Classifier")
    
#     # Save encoder
#     with open(f"{save_dir}/suit_encoder.pkl", 'wb') as f:
#         pickle.dump(processor.suit_encoder, f)
    
#     print("\nTraining complete! Model and encoder saved.")
#     print(f"Suit Classifier Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

# if __name__ == "__main__":
#     main()




import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

class CardDataProcessor:
    def __init__(self, image_size=(50, 50), threshold=128):
        self.image_size = image_size
        self.threshold = threshold
        self.number_encoder = LabelEncoder()
        
    def preprocess_image(self, image_path):
        """Load, resize and binarize image"""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.image_size)
        return img.flatten() / 255.0  # Flatten and normalize
    
    def load_data(self, data_dir):
        """Load number images"""
        images = []
        labels = []
        data_path = Path(data_dir)
        
        print("\nLoading number images...")
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                print(f"Processing class: {class_dir.name}")
                count = 0
                for img_path in class_dir.glob('*.*'):
                    images.append(self.preprocess_image(img_path))
                    labels.append(class_dir.name)
                    count += 1
                print(f"  Found {count} images")
        
        return np.array(images), labels
    
    def prepare_data(self, data_dir):
        """Prepare number dataset"""
        X_numbers, y_numbers = self.load_data(data_dir)
        y_numbers_encoded = self.number_encoder.fit_transform(y_numbers)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_numbers, y_numbers_encoded, test_size=0.2, random_state=42)
        
        return X_train, X_val, y_train, y_val

    def visualize_samples(self, X, y, title):
        """Visualize sample images from dataset"""
        plt.figure(figsize=(15, 5))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            img = X[i].reshape(self.image_size)
            plt.imshow(img, cmap='gray')
            plt.title(f'Class: {self.number_encoder.inverse_transform([y[i]])[0]}')
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

def create_improved_mlp_model(input_shape, num_classes):
    """Create an improved MLP model"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Add batch normalization
        layers.BatchNormalization(),
        
        # First dense block
        layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Second dense block
        layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Third dense block
        layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a simpler optimizer setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history, title):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, encoder, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title(f'{title} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Configuration
    IMAGE_SIZE = (50, 50)
    THRESHOLD = 128
    EPOCHS = 100
    PROBLEM_NUMBERS = ['3', '4', '5', '6']  # Problematic numbers
    
    # Get data directory
    data_dir = input("Enter the path to training data directory: ")
    save_dir = input("Enter the path to save model and encoder: ")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize processor and load data
    processor = CardDataProcessor(IMAGE_SIZE, THRESHOLD)
    X_train, X_val, y_train, y_val = processor.prepare_data(data_dir)
    
    # Create sample weights for problematic numbers
    sample_weights = np.ones(len(y_train))
    for num in PROBLEM_NUMBERS:
        num_idx = processor.number_encoder.transform([num])[0]
        sample_weights[y_train == num_idx] = 2.0  # Double weight for problematic numbers
    
    # Visualize sample images
    processor.visualize_samples(X_train, y_train, "Sample Number Images")
    
    # Create and train model
    print("\nTraining number classifier...")
    model = create_improved_mlp_model(IMAGE_SIZE[0] * IMAGE_SIZE[1], 
                                    len(processor.number_encoder.classes_))
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', 
                         factor=0.2, 
                         patience=5, 
                         min_lr=1e-6,
                         verbose=1),
        EarlyStopping(patience=20, 
                     restore_best_weights=True,
                     verbose=1),
        ModelCheckpoint(f"{save_dir}/number_model_best.h5", 
                       save_best_only=True,
                       verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        sample_weight=sample_weights,
        callbacks=callbacks,
        batch_size=32,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, "Number Classifier")
    
    # Generate predictions and plot confusion matrix
    y_pred = model.predict(X_val).argmax(axis=1)
    plot_confusion_matrix(y_val, y_pred, processor.number_encoder, "Number Classifier")
    
    # Save encoder
    with open(f"{save_dir}/number_encoder.pkl", 'wb') as f:
        pickle.dump(processor.number_encoder, f)
    
    # Print final metrics
    print("\nTraining complete! Model and encoder saved.")
    print(f"Number Classifier Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Print performance on problematic numbers
    print("\nPerformance on problematic numbers:")
    for num in PROBLEM_NUMBERS:
        num_idx = processor.number_encoder.transform([num])[0]
        num_mask = y_val == num_idx
        if np.any(num_mask):
            num_acc = (y_pred[num_mask] == y_val[num_mask]).mean()
            print(f"Number {num} accuracy: {num_acc:.4f}")

if __name__ == "__main__":
    main()