# Libraries imported
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load the training and testing datasets
instagram_df_train = pd.read_csv('insta_train.csv')
instagram_df_test = pd.read_csv('insta_test.csv')

# EDA - Summary and Info
print(instagram_df_train.info())
print(instagram_df_train.describe())

# Checking for null values
print(instagram_df_train.isnull().sum())

# EDA Visualizations with improved clarity
sns.countplot(instagram_df_train['fake'])
plt.title('Distribution of Fake Profiles in Training Set')
plt.show()

sns.countplot(instagram_df_train['private'])
plt.title('Private Profiles in Training Set')
plt.show()

plt.figure(figsize=(10,6))
sns.distplot(instagram_df_train['nums/length username'], bins=30)
plt.title('Distribution of Username Length Ratio')
plt.show()

# Correlation Matrix Plot
plt.figure(figsize=(15,10))
cm = instagram_df_train.corr()
sns.heatmap(cm, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# Data Preprocessing
X_train = instagram_df_train.drop(columns=['fake'])
X_test = instagram_df_test.drop(columns=['fake'])
y_train = instagram_df_train['fake']
y_test = instagram_df_test['fake']

# Using MinMaxScaler for better feature scaling (especially for binary features)
scaler_x = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Converting target variables into categorical form
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Improved Model with Early Stopping and Batch Normalization
model = Sequential()

model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Adding Early Stopping to prevent overfitting
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model Training
epochs_hist = model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1, validation_split=0.2, callbacks=[early_stopping_monitor])

# Plotting Losses
plt.plot(epochs_hist.history['loss'], label='Training Loss')
plt.plot(epochs_hist.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Losses')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Model Predictions and Performance Evaluation
predicted = model.predict(X_test)
predicted_value = np.argmax(predicted, axis=1)
test = np.argmax(y_test, axis=1)

print(classification_report(test, predicted_value))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test, predicted_value)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC-AUC Curve
fpr, tpr, thresholds = roc_curve(test, predicted[:,1])
roc_auc = roc_auc_score(test, predicted[:,1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='best')
plt.show()

# Save the model
model.save('instagram_fake_profile_model.h5')
