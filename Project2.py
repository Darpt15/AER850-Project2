import os

# Method 1: Change directory to where your data is
os.chdir('/Users/unnatipatel/Desktop/TMU Files/AER850 - Intro to Machine Learning/AER850-Project2/Data')
# Print current working directory
print("Current directory:", os.getcwd())

import os
def verify_dataset():
    # Paths
    base_dirs = ['Train', 'Valid', 'Test']
    classes = ['crack', 'missing-head', 'paint-off']
    
    for base_dir in base_dirs:
        print(f"\nChecking {base_dir} directory:")
        total_images = 0
        
        for class_name in classes:
            path = os.path.join(base_dir, class_name)
            if not os.path.exists(path):
                print(f"ERROR: Missing directory {path}")
                continue
                
            n_images = len([f for f in os.listdir(path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{class_name}: {n_images} images")
            total_images += n_images
            
        print(f"Total images in {base_dir}: {total_images}")

if __name__ == "__main__":
    verify_dataset()
  
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from tensorflow.keras import layers, models

import sys
print("Python path:", sys.executable)

# Verify packages
try:
    import matplotlib
    print("Matplotlib version:", matplotlib.__version__)
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    import numpy as np
    print("NumPy version:", np.__version__)
    import pandas as pd
    print("Pandas version:", pd.__version__)
    from PIL import Image
    print("Pillow is installed successfully")
    print("Pillow version:", Image.__version__)
except ImportError as e:
    print("Error importing:", e)
    
# Check installed packages
!pip list | grep -i pil

# Define constants as per requirements
IMG_HEIGHT = 500  # As per requirements
IMG_WIDTH = 500   # As per requirements
CHANNELS = 3
BATCH_SIZE = 32

def create_data_generators():
    train_dir = 'Train'
    valid_dir = 'Valid'
    
    # Define class names and create label mapping
    class_names = ['crack', 'missing-head', 'paint-off']
    label_map = {name: i for i, name in enumerate(class_names)}
    
    def process_path(file_path):
        # Extract label from path
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]  # Class name is second-to-last part of path
        
        # Convert label name to integer
        label = tf.argmax(label == class_names)
        label = tf.one_hot(label, len(class_names))
        
        # Read and process image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, label
    
    # Create train dataset
    train_ds = tf.data.Dataset.list_files(str(train_dir + '/*/*.jpg'))
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    valid_ds = tf.data.Dataset.list_files(str(valid_dir + '/*/*.jpg'))
    valid_ds = valid_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(BATCH_SIZE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, valid_ds

def create_model_with_hyperparameters(
    conv_activation='relu',
    dense_activation='relu',
    conv_filters=[32, 64, 128, 256],
    dense_neurons=[512, 256],
    optimizer='adam',
    dropout_rates=[0.5, 0.3]
):
    """
    Create CNN with configurable hyperparameters
    
    Parameters:
    - conv_activation: Activation function for conv layers ('relu' or 'leaky_relu')
    - dense_activation: Activation function for dense layers ('relu' or 'elu')
    - conv_filters: List of filters for conv layers (powers of 2)
    - dense_neurons: List of neurons for dense layers
    - optimizer: Optimizer for model compilation
    - dropout_rates: List of dropout rates for dense layers
    """
    
    # Configure activation functions
    if conv_activation == 'leaky_relu':
        conv_act = tf.keras.layers.LeakyReLU(alpha=0.01)
    else:
        conv_act = 'relu'
        
    model = models.Sequential()
    
    # First Convolutional Block
    model.add(layers.Conv2D(
        conv_filters[0], 
        kernel_size=(3, 3), 
        activation=conv_act,
        input_shape=(500, 500, 3), 
        padding='same'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Additional Convolutional Blocks
    for filters in conv_filters[1:]:
        model.add(layers.Conv2D(
            filters, 
            kernel_size=(3, 3), 
            activation=conv_act,
            padding='same'
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten layer
    model.add(layers.Flatten())
    
    # Dense layers
    for neurons, dropout_rate in zip(dense_neurons, dropout_rates):
        model.add(layers.Dense(neurons, activation=dense_activation))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer with softmax activation
    model.add(layers.Dense(3, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to experiment with different hyperparameters
def experiment_with_hyperparameters():
    # Define different hyperparameter combinations
    experiments = [
        {
            'name': 'Base Model (ReLU)',
            'params': {
                'conv_activation': 'relu',
                'dense_activation': 'relu',
                'conv_filters': [32, 64, 128, 256],
                'dense_neurons': [512, 256],
                'optimizer': 'adam',
                'dropout_rates': [0.5, 0.3]
            }
        },
        {
            'name': 'LeakyReLU Model',
            'params': {
                'conv_activation': 'leaky_relu',
                'dense_activation': 'relu',
                'conv_filters': [32, 64, 128, 256],
                'dense_neurons': [512, 256],
                'optimizer': 'adam',
                'dropout_rates': [0.5, 0.3]
            }
        },
        {
            'name': 'ELU Dense Model',
            'params': {
                'conv_activation': 'relu',
                'dense_activation': 'elu',
                'conv_filters': [32, 64, 128, 256],
                'dense_neurons': [512, 256],
                'optimizer': 'adam',
                'dropout_rates': [0.5, 0.3]
            }
        }
    ]
    
    models = {}
    for exp in experiments:
        print(f"\nCreating {exp['name']}:")
        model = create_model_with_hyperparameters(**exp['params'])
        model.summary()
        models[exp['name']] = model
    
    return models
    

import matplotlib.pyplot as plt

def train_and_evaluate_model(model, train_generator, validation_generator, epochs=30):
    """
    Train the model and return its history
    """
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    return history

def plot_training_history(history, title="Model Performance"):
    """
    Plot training & validation accuracy and loss
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    # Print final values
    print("\nFinal Training Metrics:")
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")

def evaluate_models(models_dict, train_generator, validation_generator):
    """
    Evaluate multiple models and compare their performance
    """
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\nTraining and evaluating {model_name}...")
        
        # Train model
        history = train_and_evaluate_model(model, train_generator, validation_generator)
        
        # Plot results
        plot_training_history(history, title=f"{model_name} Performance")
        
        # Store results
        results[model_name] = {
            'train_acc': history.history['accuracy'][-1],
            'val_acc': history.history['val_accuracy'][-1],
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1]
        }
    
    # Compare models
    print("\nModel Comparison:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Training Accuracy: {metrics['train_acc']:.4f}")
        print(f"Validation Accuracy: {metrics['val_acc']:.4f}")
        print(f"Training Loss: {metrics['train_loss']:.4f}")
        print(f"Validation Loss: {metrics['val_loss']:.4f}")

# Usage example
if __name__ == "__main__":
    # Assuming you have your data generators and models ready
    train_generator, validation_generator = create_data_generators()
    models = experiment_with_hyperparameters()
    
    # Evaluate all models
    evaluate_models(models, train_generator, validation_generator)
    
    # Save the model
    best_model.save('best_model.h5')
    
    

    
    