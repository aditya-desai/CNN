from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import cv2
import os
from tqdm import tqdm
import numpy as np
from random import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = 'dogsvscats.model'

TRAIN_DIR = 'D:/Data/Cats V Dogs/train'

# Assign labels and convert them into one-hot arrays
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]

# Preprocess the training and test data
def create_data(dir):
    training_data = []
    for img in tqdm(os.listdir(dir)):
        label = label_img(img)
        path = os.path.join(dir,img)
        img = cv2.imread(path)
        g = cv2.resize(img, (32, 32, 3))
        training_data.append([np.array(g), np.array(labels)])
        random.shuffle(training_data)
        np.save('train_data.npy', training_data)
    return training_data 

images = []
for img in os.listdir('D:/Data/Cats V Dogs/test'):
    path = os.path.join('D:/Data/Cats V Dogs/test',img)
    img = cv2.imread(path)
    images.append(img)

train_data = create_train_data(TRAIN_DIR)

print(np.shape(train_data))
train = train_data[:-500]
test = train_data[-500:]

train_x = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
train_y = np.array([i[1] for i in train])
print(np.shape(train_x))

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = np.array([i[1] for i in test])
print(np.shape(train_y))

X = train_x
Y = train_y
X_test = test_x
Y_test = test_y

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation = 'relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation = 'relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)

#if os.path.exists('{}.meta'.format(MODEL_NAME)):
#    model.load(MODEL_NAME)
#    print('model loaded!')

model.fit(X, Y, n_epoch = 10, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='catsvdogs_cnn')
print('test')
model.save(MODEL_NAME)

model_out = np.array(model.predict(test_x))
predi = []
for row in model_out:
    predi.append(np.argmax(row))
predi = np.array(predi)
def sort_wrong_images():
    
    predy = []
    for row in test_y:
        predy.append(np.argmax(row))
    predy = np.array(predy)
    wrong = predi - predy
    accrcheck = wrong
    accrcheck = np.array(accrcheck)
    for _ in range(600):
       if accrcheck[_] != 0:
            accrcheck[_] = 1
    wrong_imageind = []
    correct_imageind = []
    for row in range(600):
        if wrong[row] != 0:
            wrong_imageind.append(row)
        else:
            correct_imageind.append(row)
    list(wrong_imageind)
    return _, accrcheck, predy, row, wrong, wrong_imageind, correct_imageind

_, accrcheck, predy, row, wrong, wrong_imageind, correct_imageind = sort_wrong_images()

avrl = 0
for ind in wrong_imageind:
    #img = np.reshape(test_x[ind], (32, 32, 3))
    if predi[ind] == 0:
        name = 'cat'
    else:
        name = 'dog'
    cv2.imwrite('D:/Data/Cats V Dogs/Wrong/{} ({}).png'.format(name, avrl), images[ind])
    avrl += 1

avrl = 0
for ind in correct_imageind:
    #img = np.reshape(test_x[ind], (32, 32, 3))
    if predi[ind] == 0:
        name = 'cat'
    else:
        name = 'dog'
    cv2.imwrite('D:/Data/Cats V Dogs/Correct/{} ({}).png'.format(name, avrl), images[ind])
    avrl += 1
