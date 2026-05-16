import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Simulate Dummy Image Data (Dataset of 100 images, size 64x64, with 3 channels RGB)
X_train = np.random.random((100, 64, 64, 3))
y_train = np.random.randint(0, 10, size=(100,)) # 10 different image classes (0-9)

# 2. Build the Convolutional Neural Network (CNN) Architecture
def build_cnn_model():
    model = models.Sequential([
        # 1st Convolutional Layer + MaxPooling
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # 2nd Convolutional Layer + MaxPooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flattening and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax') # Softmax output for multi-class classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Initialize and summary of the model
model = build_cnn_model()
print("Project 02: Image Classification Using CNN")
print("------------------------------------------")
model.summary()

# 4. Fit model on dummy data to ensure it runs without errors
model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1)
print("\nCNN Model Built and Trained Successfully!")
          
