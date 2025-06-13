import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import random
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

DATASET_PATH = r"D:\Internship_Tasks\Maharshi_Animal-_and_Human_Detection_Model\Dataset"  
OUTPUT_DIR = "extractedd_frames"  
IMAGE_SIZE = (224, 224)  
FRAMES_PER_VIDEO = 30  
TEST_SPLIT = 0.2  
VAL_SPLIT = 0.2  
BATCH_SIZE = 16
EPOCHS = 10

def create_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ['train', 'val', 'test']:
        for class_name in os.listdir(DATASET_PATH):
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

def extract_frames():
    """Extract frames from videos and save them as images."""
    print("extracting frames from videos...")
    
    video_files = []
    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                class_name = os.path.basename(root)
                video_files.append((os.path.join(root, file), class_name))
    
    for video_path, class_name in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(OUTPUT_DIR, 'all_frames', class_name, video_name)
        
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= FRAMES_PER_VIDEO:
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < FRAMES_PER_VIDEO:
            frame_step = 1
        else:
            frame_step = total_frames // FRAMES_PER_VIDEO
        
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened() and saved_count < FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_step == 0:
                frame = cv2.resize(frame, IMAGE_SIZE)
                frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                
                if saved_count >= FRAMES_PER_VIDEO:
                    break
                    
            frame_count += 1
            
        cap.release()

def split_dataset():
    print("Splitting dataset...")
    
    frame_paths = []
    labels = []
    class_names = sorted(os.listdir(os.path.join(OUTPUT_DIR, 'all_frames')))
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(OUTPUT_DIR, 'all_frames', class_name)
        for video_dir in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_dir)
            for frame_file in os.listdir(video_path):
                if frame_file.endswith('.jpg'):
                    frame_paths.append(os.path.join(video_path, frame_file))
                    labels.append(class_idx)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        frame_paths, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SPLIT, random_state=42, stratify=y_train_val
    )
    
    def move_files(file_paths, split_name):
        for file_path in tqdm(file_paths, desc=f"Moving {split_name} files"):
            class_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            file_name = os.path.basename(file_path)
            dest_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(file_path, os.path.join(dest_dir, file_name))
    
    move_files(X_train, 'train')
    move_files(X_val, 'val')
    move_files(X_test, 'test')

def create_data_generators():
    """create data generators for training validation and testing."""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, 'train'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = test_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, 'val'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, 'test'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def build_model(num_classes):
    """build and compile the model."""
    #use mobilenetv2 as base model
    base_model = applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    
    base_model.trainable = False
    
    #customizing base model
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    print("Training model...")
    
    train_generator, val_generator, test_generator = create_data_generators()
    
    model = build_model(num_classes=len(train_generator.class_indices))
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    model.save('animal_person_classifier.h5')
    
    plot_training_history(history)
    
    return model, test_accuracy

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    
    create_directories()
    
    extract_frames()
    
    split_dataset()
    
    model, test_accuracy = train_model()
    
    print(f"\nTraining complete! Model saved with test accuracy: {test_accuracy:.4f}")
    print("model saved as 'animal_person_classifier.h5'")
    print("training history plot saved as 'training_history.png'")

if __name__ == "__main__":
    main()
