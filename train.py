#Importing the required packages..
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf
#Importing tensorflow 2.6
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
#Reading data from the gcs bucket
dataset = pd.read_csv(r"gs://vertex-ai-custom/CrabAgePrediction.csv")
dataset.tail()
BUCKET = 'gs://vertex-ai-123-bucket'
dataset.isna().sum()
dataset = dataset.dropna()
#Data transformation..
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()
#Dataset splitting..
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_stats = train_dataset.describe()
#Removing age column, since it is a target column
train_stats.pop("Age")
train_stats = train_stats.transpose()
train_stats
#Removing age column from train and test data
train_labels = train_dataset.pop('Age')
test_labels = test_dataset.pop('Age')
def norma_data(x):
    #To normalise the numercial values
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norma_data(train_dataset)
normed_test_data = norma_data(test_dataset)
def build_model():
    #model building function
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

#model = build_model()

#model.summary()

model = build_model()
EPOCHS = 10

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels,
                    epochs=EPOCHS, validation_split = 0.2,
                    callbacks=[early_stop])
model.save(BUCKET + '/model')