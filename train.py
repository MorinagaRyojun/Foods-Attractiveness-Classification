from __future__ import absolute_import, division, print_function

import tensorflow as tf




from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

from tensorflow.keras import models
from tensorflow.keras.applications.inception_v3 import preprocess_input


import os

import numpy as np
# from scipy.misc import imresize
# from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


root_position = './input_data/clean/'

food_list = ['Best','Worst']
src_train = root_position+'train'
src_test = root_position+'test'

print("Total number of samples in train folder")
train_files = sum([len(files) for i, j, files in os.walk(src_train)])
print(train_files)

print("Total number of samples in test folder")
test_files = sum([len(files) for i, j, files in os.walk(src_test)])
print(test_files)

def train_model(n_classes,num_epochs, nb_train_samples,nb_validation_samples):
  tf.keras.backend.clear_session()

  img_width, img_height = 299, 299
  train_data_dir = src_train
  validation_data_dir = src_test
  batch_size = 16
  bestmodel_path = root_position+'weights-improvement-{epoch:02d}-{val_accuracy:.2f} bestmodel_2class.hdf5'
  trainedmodel_path = root_position+'weights-improvement-{epoch:02d}-{val_accuracy:.2f} trainedmodel_2class.hdf5'
  history_path = root_position+'history_'+str(n_classes)+'.log'

  train_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)

  test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

  train_generator = train_datagen.flow_from_directory(
      train_data_dir,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical')

  validation_generator = test_datagen.flow_from_directory(
      validation_data_dir,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical')


  inception = InceptionV3(weights='imagenet', include_top=False)
  x = inception.output
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.2)(x)

  predictions = Dense(n_classes,kernel_regularizer=tf.keras.regularizers.L2(0.005), activation='softmax')(x)

  model = Model(inputs=inception.input, outputs=predictions)
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
  checkpoint = ModelCheckpoint(filepath=bestmodel_path, verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')
  csv_logger = CSVLogger(history_path)

  history = model.fit_generator(train_generator,
                      steps_per_epoch = nb_train_samples // batch_size,
                      validation_data=validation_generator,
                      validation_steps=nb_validation_samples // batch_size,
                      epochs=num_epochs,
                      verbose=1,
                      callbacks=[csv_logger, checkpoint])

  model.save(trainedmodel_path)
  class_map = train_generator.class_indices
  return history, class_map

# Train the model with data from 2 classes
n_classes = 2
epochs = 5000
nb_train_samples = train_files
nb_validation_samples = test_files

history, class_map_3 = train_model(n_classes,epochs, nb_train_samples,nb_validation_samples)
print(class_map_3)


