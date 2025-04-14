import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = 'dataset'
WEATHER_CLASSES = ['rain', 'fog', 'night', 'day']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

def load_and_preprocess_data():
    images = []
    labels = []
    class_to_idx = {cls: idx for idx, cls in enumerate(WEATHER_CLASSES)}

    for weather_class in WEATHER_CLASSES:
        class_dir = os.path.join(DATASET_DIR, weather_class)
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist!")
            continue

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                images.append(img_array)
                labels.append(class_to_idx[weather_class])
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
                continue

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_metrics(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax1.set_title('Accuracy Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_title('Loss Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Training metrics plot saved as 'training_metrics.png'")

def evaluate_and_plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=WEATHER_CLASSES))

    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=WEATHER_CLASSES, yticklabels=WEATHER_CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Weather')
    plt.ylabel('True Weather')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

def main():
    print("Loading dataset...")
    images, labels = load_and_preprocess_data()
    print(f"Loaded {len(images)} images with shape {images.shape}")

    X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")

    print("Building model...")
    model = build_model(num_classes=len(WEATHER_CLASSES))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_weather_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    plot_training_metrics(history)
    evaluate_and_plot_confusion_matrix(model, X_test, y_test)

    model.save('final_weather_model.h5')
    print("Final model saved as 'final_weather_model.h5'")

if __name__ == '__main__':
    main()