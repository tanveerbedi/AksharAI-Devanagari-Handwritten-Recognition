# train_model.py (with Data Augmentation)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras import layers
import os

print("TensorFlow Version:", tf.__version__)

def train_and_save_model():
    """
    Loads the Devanagari dataset, applies data augmentation, builds a CNN,
    trains it, and saves the model.
    """
    # --- 1. Load Dataset Efficiently from Local Folder ---
    print("Loading Devanagari Handwritten Character Dataset using tf.data pipeline...")
    
    local_dataset_path = 'DevanagariHandwrittenCharacterDataset'
    data_dir = os.path.join(local_dataset_path, 'Train')

    if not os.path.isdir(data_dir):
        print(f"Error: Dataset folder not found at '{data_dir}'")
        print("Please download the dataset from Kaggle, unzip it, and place the")
        print(f"'{local_dataset_path}' folder in the same directory as this script.")
        return

    BATCH_SIZE = 128
    IMG_SIZE = (32, 32)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # --- 2. Create the Data Augmentation Layer ---
    # This layer will randomly alter the training images to make the model more robust.
    data_augmentation = Sequential(
      [
        layers.RandomRotation(0.1), # Randomly rotate images by up to 10%
        layers.RandomZoom(0.1),   # Randomly zoom in on images by up to 10%
      ],
      name="data_augmentation",
    )

    # --- 3. Build the CNN Model with Augmentation ---
    print("Building the CNN model with data augmentation...")
    model = Sequential([
        # Add the data augmentation layer. It will only be active during training.
        data_augmentation,
        
        # The rest of the model is the same.
        Rescaling(1./255, input_shape=(32, 32, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- 4. Train the Model ---
    print("Training the model...")
    # The data augmentation layer is automatically applied during model.fit()
    model.fit(train_ds, epochs=15, validation_data=val_ds)

    # --- 5. Evaluate and Save ---
    loss, accuracy = model.evaluate(val_ds)
    print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
    print("Saving the trained model to 'devanagari_model.h5'...")
    model.save('devanagari_model.h5')
    print("Model saved successfully!")

if __name__ == '__main__':
    train_and_save_model()
