# 🎤 SPEECH EMOTION RECOGNITION - Full Python Script
# ---------------------------------------------------

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------
# STEP 1: Load and visualize a sample audio file
# ---------------------------------------------
DATA_PATH = "ravdess_data/"  # change this path as needed
file_path = os.path.join(DATA_PATH, "03-01-01-01-01-01-01.wav")

signal, sr = librosa.load(file_path, sr=22050)
print(f"Audio duration: {librosa.get_duration(y=signal, sr=sr):.2f} seconds")

plt.figure(figsize=(10, 4))
librosa.display.waveshow(signal, sr=sr)
plt.title("Raw Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# STEP 2: Extract MFCC Features
# ---------------------------------------------
mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
print("MFCCs shape:", mfccs.shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title("MFCC (Mel Frequency Cepstral Coefficients)")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# STEP 3: Feature Extraction Function
# ---------------------------------------------
def extract_features(file_path, n_mfcc=13):
    signal, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# ---------------------------------------------
# STEP 4: Load Dataset and Extract Features
# ---------------------------------------------
features = []
labels = []

for file_name in os.listdir(DATA_PATH):
    if file_name.endswith(".wav"):
        emotion_label = file_name.split("-")[2]  # extract emotion code from filename
        file_path = os.path.join(DATA_PATH, file_name)
        mfcc = extract_features(file_path)
        features.append(mfcc)
        labels.append(emotion_label)

X = np.array(features)
y = np.array(labels)

# ---------------------------------------------
# STEP 5: Train-Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------
# STEP 6: Model Training (Random Forest + SVM)
# ---------------------------------------------
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
svm_params = {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.1]}

cv = StratifiedKFold(n_splits=10)
rf_grid = GridSearchCV(rf_model, rf_params, cv=cv, scoring='accuracy', n_jobs=-1)
svm_grid = GridSearchCV(svm_model, svm_params, cv=cv, scoring='accuracy', n_jobs=-1)

rf_grid.fit(X_train, y_train)
svm_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
best_svm = svm_grid.best_estimator_

# ---------------------------------------------
# STEP 7: Evaluate Models
# ---------------------------------------------
rf_preds = best_rf.predict(X_test)
svm_preds = best_svm.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
svm_acc = accuracy_score(y_test, svm_preds)

print("✅ Random Forest Accuracy:", rf_acc)
print("✅ SVM Accuracy:", svm_acc)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, rf_preds), display_labels=best_rf.classes_).plot(ax=axes[0], cmap="Greens")
axes[0].set_title("Random Forest Confusion Matrix")

ConfusionMatrixDisplay(confusion_matrix(y_test, svm_preds), display_labels=best_svm.classes_).plot(ax=axes[1], cmap="Blues")
axes[1].set_title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# STEP 8: Save the Best Model
# ---------------------------------------------
joblib.dump(best_rf, "speech_emotion_model.pkl")
print("🎯 Model saved as 'speech_emotion_model.pkl'")

# ---------------------------------------------
# STEP 9: CNN Model (Alternative Deep Learning)
# ---------------------------------------------
def build_cnn_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------------------------
# STEP 10: Per-Class Accuracy Plot Function
# ---------------------------------------------
def plot_per_class_accuracy(y_true, y_pred, labels):
    accuracies = {}
    for label in labels:
        indices = [i for i, y in enumerate(y_true) if y == label]
        if len(indices) == 0:
            accuracies[label] = 0
            continue
        sub_y_true = [y_true[i] for i in indices]
        sub_y_pred = [y_pred[i] for i in indices]
        accuracies[label] = accuracy_score(sub_y_true, sub_y_pred)

    df = pd.DataFrame(list(accuracies.items()), columns=['Emotion', 'Accuracy'])
    plt.figure(figsize=(8, 5))
    plt.bar(df['Emotion'], df['Accuracy'], color='skyblue', edgecolor='black')
    plt.ylim(0, 1)
    plt.title('Per-Emotion Accuracy')
    plt.xlabel('Emotion')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Example dummy test:
y_test_example = [random.choice(['angry', 'calm', 'happy', 'sad']) for _ in range(50)]
y_pred_example = [random.choice(['angry', 'calm', 'happy', 'sad']) for _ in range(50)]
plot_per_class_accuracy(y_test_example, y_pred_example, ['angry', 'calm', 'happy', 'sad'])
