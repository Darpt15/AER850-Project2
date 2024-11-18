import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def test_model(model_path, test_images):
    model = tf.keras.models.load_model(model_path)
    class_names = ['crack', 'missing-head', 'paint-off']

    for img_path, true_class in test_images:
        img = image.load_img(img_path, target_size=(500, 500))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'Test Image\nTrue Class: {true_class}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        bars = plt.bar(class_names, predictions[0])
        bars[predicted_class_idx].set_color('red')
        plt.title('Class Probabilities')
        plt.xlabel('Class')
        plt.ylabel('Probability')

        for j, prob in enumerate(predictions[0]):
            plt.text(j, prob, f'{prob:.2%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'prediction_{true_class}.png')
        plt.close()

        print(f"\nResults for {true_class}:")
        print(f"True Class: {true_class}")
        print(f"Predicted Class: {predicted_class}")
        print("Class Probabilities:")
        for class_name, prob in zip(class_names, predictions[0]):
            print(f"{class_name}: {prob:.2%}")

if __name__ == "__main__":
    test_images = [
        ("test/crack/test_crack.jpg", "crack"),
        ("test/missing-head/test_missinghead.jpg", "missing-head"),
        ("test/paint-off/test_paintoff.jpg", "paint-off")
    ]
    
    model_path = 'best_model.h5'  # Path to your saved model
    test_model(model_path, test_images)
    print("\nModel testing completed. Prediction images saved.")