# Import Libraries & Packages :
import os
import itertools
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid') 

# Ignore warnings 
import warnings
warnings.filterwarnings('ignore')  

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Dense , Dropout , BatchNormalization

# print("Modules Loaded")

# Data Preprocessing
# Reading The Data :
data_dir = '/kaggle/input/multiclass-weather-dataset/Multi-class Weather Dataset' 
file_paths = []
labels = []
folds = os.listdir(data_dir)
# print(folds)

for fold in folds:
    primary_sub_fold_path = os.path.join(data_dir,fold)
    # print(primary_sub_fold_path)
    secondary_sub_fold_names = os.listdir(primary_sub_fold_path)
    # print(secondary_sub_fold_names)
    for pics in secondary_sub_fold_names:
        secondary_sub_fold_path = os.path.join(primary_sub_fold_path,pics)
        # print(secondary_sub_fold_path)
        file_paths.append(secondary_sub_fold_path)
        # print(file_paths)
        labels.append(fold)
        # print(labels)
F_Series = pd.Series(file_paths , name = 'File _Series')      
L_Series = pd.Series(labels , name = 'Labels')
data_frame = pd.concat([F_Series , L_Series] , axis = 1)       

# print(data_frame) 

# Splitting Data Into (train , valid , test)
train_data_frame , test_data_frame = train_test_split(data_frame , test_size = 0.2 ,random_state = 42 , stratify = data_frame['Labels'])
valid_data_frame , test_data_frame = train_test_split(test_data_frame , test_size = 0.5 , random_state = 42 , stratify = test_data_frame['Labels'])

# print(train_data_frame)
# print(test_data_frame)
# print(valid_data_frame)

# Create Image Data Generator:
batch_size = 16
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
img_data_gen = ImageDataGenerator()
train_data_gen = img_data_gen.flow_from_dataframe(train_data_frame,x_col='File _Series',y_col='Labels',target_size=img_size,class_mode='categorical',color_mode='rgb',shuffle=True,batch_size=batch_size)
valid_data_gen = img_data_gen.flow_from_dataframe(valid_data_frame,x_col='File _Series',y_col='Labels',target_size=img_size,class_mode='categorical',color_mode='rgb',shuffle=True,batch_size=batch_size)
test_data_gen = img_data_gen.flow_from_dataframe(test_data_frame,x_col='File _Series',y_col='Labels',target_size=img_size,class_mode='categorical',color_mode='rgb',shuffle=False,batch_size=batch_size)

# Function to display sample images from the training set
def show_sample_from_training_set(data_gen, class_indices, sample_size=5):
    # Get a batch of images and labels from the training data generator
    images, labels = next(data_gen)

    # Create the plot
    plt.figure(figsize=(10, 10))

    # Loop through the number of samples to display
    for i in range(sample_size):
        ax = plt.subplot(1, sample_size, i + 1)

        # Get the image
        img = images[i]

        # Get the label (class)
        label = np.argmax(labels[i])  # Get the actual class
        class_name = list(class_indices.keys())[list(class_indices.values()).index(label)]  # Map the class to its name

        # Display the image with the class label
        plt.imshow(img.astype('uint8'))
        plt.title(class_name)
        plt.axis("off")

    plt.show()

# Show a sample of 5 images from the training set
show_sample_from_training_set(train_data_gen, train_data_gen.class_indices, sample_size=5)

# Model Structure:
base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet', input_shape=img_shape)

# Unfreeze some layers
for layer in base_model.layers[:-50]:  # Keep the first 50 layers frozen
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # L2 regularization
    Dropout(rate=0.5),  # Increased dropout
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
hist = model.fit(train_data_gen, epochs=10, verbose=1, validation_data=valid_data_gen)

# Define Needed Variables:
tr_acc = hist.history['accuracy']
tr_loss = hist.history['loss']
val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]

index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]

Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'Best Epoch = {str(index_loss + 1)}'
acc_label = f'Best Epoch = {str(index_acc + 1)}'

# Plotting training history:
plt.figure(figsize=(20, 8))
plt.style.use('fivethirtyeight')
plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training Loss')
plt.plot(Epochs, val_loss, 'g', label='Validation Loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Model evaluation
train_score = model.evaluate(train_data_gen)
valid_score = model.evaluate(valid_data_gen)
test_score = model.evaluate(test_data_gen)

# Predictions
preds = model.predict(test_data_gen)
y_pred = np.argmax(preds, axis=1)

gin_dict = test_data_gen.class_indices
classes = list(gin_dict.keys())

# Confusion Matrix
cm = confusion_matrix(test_data_gen.classes, y_pred)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max()/2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Save the model
model.save('weather_conditions_efficientnetb3.h5')