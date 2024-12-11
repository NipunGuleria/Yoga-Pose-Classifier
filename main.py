import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
data_path = "YogaPoses"  


def load_images_and_labels(data_path):
    
    images = []
    labels = []
    classes = sorted(os.listdir(data_path))  
    
    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue  # Skip unreadable files
            
            # Resize image to a consistent size (e.g., 224x224)
            image = cv2.resize(image, (224, 224))
            
            images.append(image)
            labels.append(class_index)  # Use folder index as label
            
    return np.array(images), np.array(labels), classes


def extract_keypoints(image):

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        ).flatten()
    else:
        keypoints = np.zeros(33 * 3)  
    return keypoints

def build_model(input_dim, num_classes):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # First Dense Block
    x = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Second Dense Block
    x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Third Dense Block
    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Residual Block
    shortcut = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)  # Shortcut connection
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])  # Residual addition
    x = tf.keras.layers.Activation('relu')(x)

    # Fourth Dense Block
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU(alpha=1.0)(x)  # Exponential Linear Unit for smoother gradients
    x = tf.keras.layers.Dropout(0.3)(x)

    # Fifth Dense Block
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Output Layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Model
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Fine-tuned learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
    
    # Compile the model with a learning rate scheduler
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(X_train, y_train, X_test, y_test, num_classes):
    model = build_model(X_train.shape[1], num_classes)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
    return model

def provide_feedback(keypoints, ideal_pose):
    """Compare detected pose keypoints to ideal pose."""
    feedback = []
    for i, (detected, ideal) in enumerate(zip(keypoints, ideal_pose)):
        deviation = np.linalg.norm(np.array(detected) - np.array(ideal))
        if deviation > 0.1:  # Threshold for misalignment
            feedback.append(f"Adjust keypoint {i}, deviation: {deviation:.2f}")
    return feedback

def main():

    images, labels,class_names = load_images_and_labels(data_path)
    keypoints = np.array([extract_keypoints(image) for image in images])
    num_classes = len(set(labels))
    
 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(keypoints, labels, test_size=0.2, random_state=42)

 
    model = train_model(X_train, y_train, X_test, y_test, num_classes)
    from sklearn.metrics import classification_report
    

# Generate predictions on the test set
    y_pred = np.argmax(model.predict(X_test), axis=1)

# Print classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    test_image = cv2.imread("test_yoga_pose.jpg") 
    test_keypoints = extract_keypoints(test_image)
    prediction = model.predict(test_keypoints.reshape(1, -1)).argmax()
    
    print(f"Predicted Yoga Pose: {class_names[prediction]}")

    ideal_pose = np.zeros_like(test_keypoints)  
    feedback = provide_feedback(test_keypoints.reshape(-1, 3), ideal_pose.reshape(-1, 3))
    print("Feedback:", feedback)

if __name__ == "__main__":
    main()
