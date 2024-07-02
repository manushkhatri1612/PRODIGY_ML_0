#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# In[2]:



dataset_dir = 'D:\Machine learning\Prodigy ML\Task-4\DATASET'  

img_size = (64, 64) 

def load_data(dataset_dir, img_size):
    data = []
    labels = []
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, img_name)
                    img = load_img(img_path, target_size=img_size, color_mode='grayscale')
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0  
                    data.append(img_array)
                    labels.append(folder)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(data), np.array(labels)

data, labels = load_data(dataset_dir, img_size)

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[4]:


from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_binarizer.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[5]:


history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[6]:


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# In[16]:


y_pred = model.predict(X_test)

def visualize_predictions(X_test, y_test, y_pred, label_binarizer, img_size, n_samples=20):
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    plt.figure(figsize=(50, 50))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(X_test[i].reshape(img_size[0], img_size[1]), cmap='gray')
        true_label = label_binarizer.classes_[y_test_labels[i]]
        pred_label = label_binarizer.classes_[y_pred_labels[i]]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.show()

visualize_predictions(X_test, y_test, y_pred, label_binarizer, img_size)

# %%
