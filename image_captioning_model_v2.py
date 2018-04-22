from __future__ import print_function

import os
import sys
import numpy as np
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Input, LSTM, Reshape, Flatten, Dropout, multiply
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
from glob import glob

import tensorflow as tf


#--------------------------------------
# Loading the data
#--------------------------------------

#Loading laebls and texts
# Labels - image names form dataset
# text - captions from dataset

#Load from already stored text files

texts = []
text_file = []
labels = []

dir_name = './Flickr8k/Flickr8k_text'

f = open(os.path.join(dir_name, 'Flickr8k.token.txt'))
text_file.append(f.read())
f.close()
texts_str = ''.join(text_file)
entries = texts_str.split("\n")

for entry in range(len(entries)-1):
    data_point = re.split('#|\t', entries[entry])
    labels.append(data_point[0])
    texts.append(data_point[2])
    # print(data_point[0])


#data limits
texts = texts[:1000]
labels = labels[:1000]

# Vectorizing the text captions

maxlen = 200  # maximum 100 words in a sequence
max_words = 10000  # We will only consider the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
  
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
# print(word_index['like'])
  
sentence_vector = pad_sequences(sequences, maxlen=maxlen)

print('Shape of data tensor:', sentence_vector.shape)



#Formulate image input

img_rows = 150
img_cols = 150
channels = 3
img_shape = (img_rows, img_cols, channels)

image_set = []

for label in range(len(labels)):
	dir_imgs = "./Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset/" + str(labels[label])
	image = Image.open(dir_imgs)
	image = image.resize([img_rows , img_cols])
	image_nparray = np.array(image.convert('RGB'))
	image_set.append(image_nparray)

image_set = np.asarray(image_set)


print("Data loading done")

print(image_set.shape)




#-----------------------------------
#  IMAGE EMBEDDING MODEL
#-----------------------------------

image_input = Input(shape=img_shape)

# create the base pre-trained model
# note the include top is set to False, meaning the last layer is eliminated
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=img_shape)

#obtain the output of the pretrained model and add custom last layer
# add a global spatial average pooling layer
image_model = base_model(image_input)
image_model = GlobalAveragePooling2D()(image_model)

# image_model = Flatten(input_shape=image_model.output_shape[1:])(image_model)

# add a fully connected layer
image_model = Dense(1024, activation='relu')(image_model)

# add a softmax layer according to the no. of classes
image_model = Dense(200, activation='softmax')(image_model)

#freeza all the layers in the pretrained model so they won't be trained
for layer in base_model.layers:
	layer.trainable = False


image_model_final = Model(image_input, image_model)

#---------------------------------
#   WORD MODEL
#---------------------------------

#feeding the output of the image model to here
image_embed_input = Input(shape=(200,))

#run it through a final LSTM layer
# encoded_sentence_output = LSTM(200)(image_embed_input)

encoded_sentence_output = Dense(200, activation='softmax')(image_embed_input)


#The word embedding model 
sentence_model_final = Model(image_embed_input , encoded_sentence_output)

#fedding the image model to the word model and obtaining the output
# final_output = sentence_model_final(image_model_final(image_input))
final_output = sentence_model_final(image_model)

# The main model. Input - image pinput. Output - word emnedding output
model = Model(image_input, final_output)
# Trainable mode

#compiling the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

#training and validation set sizes
training_samples = 200  
validation_samples = 800

# sentence_vector = np.asarray(sentence_vector, dtype=np.float32)

#dividing the data into training and validation sets
x_train = image_set[:training_samples]
y_train = sentence_vector[:training_samples]

# y_train = K.variable(np.random.randn(200))
x_val = image_set[training_samples: training_samples + validation_samples]
y_val = sentence_vector[training_samples: training_samples + validation_samples]

	


history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("accuracy: " + acc)
print("validation accuracy: " + val_acc)
print("loss: " + loss)
print("validation loss: " + val_loss)

#saving the model
model.save_weights('image_captioning_model_v2.h5')