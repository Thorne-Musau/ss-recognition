# Import Libraries 
import os
import glob
import sys
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# Function to load and extract features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=512)
    return mfccs

def load_audio_files_from_dir(directory, label):
    features = []
    labels = []
    
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            file_path = os.path.join(directory, file)
            mfccs = extract_features(file_path)
            for mfcc in mfccs.T:  # Each column is a feature vector for one frame
                features.append(mfcc)
                labels.append(label)
    
    return features, labels

# Define paths to the datasets
automatic_path = 'Gunshots (1)/Gunshots/Automatic'
single_shot_path = 'Gunshots (1)/Gunshots/Single Shot'
ambience_path = '1_Ambience Sound/Ambience Sound'

# Load gunshots data
automatic_features, automatic_labels = load_audio_files_from_dir(automatic_path, label=1)
single_shot_features, single_shot_labels = load_audio_files_from_dir(single_shot_path, label=1)

# Combine the gunshots data from both folders
gunshots_features = automatic_features + single_shot_features
gunshots_labels = automatic_labels + single_shot_labels

# Load ambience data
ambience_features, ambience_labels = load_audio_files_from_dir(ambience_path, label=0)

# Combine all features and labels
features = gunshots_features + ambience_features
labels = gunshots_labels + ambience_labels

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Stratified Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Reshape data for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Binarize labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Build the CNN model
model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
