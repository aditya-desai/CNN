from __future__ import division, print_function, absolute_import
from timeit import default_timer as timer
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



TRAIN_DIR = 'D:/Data/Fruit Dataset/Train_32'
TEST_DIR =  'D:/Data/Fruit Dataset/Test_32'

# Assign labels and convert them into one-hot arrays
def label_img(img):
    word_label = img.split(' ')[-2]
    if word_label == 'Apple': return [1,0,0,0,0,0,0,0,0,0]
    elif word_label == 'Banana': return [0,1,0,0,0,0,0,0,0,0]
    elif word_label == 'Grapes': return [0,0,1,0,0,0,0,0,0,0]
    elif word_label == 'Jackfruit': return [0,0,0,1,0,0,0,0,0,0]
    elif word_label == 'Mango': return [0,0,0,0,1,0,0,0,0,0]
    elif word_label == 'Orange': return [0,0,0,0,0,1,0,0,0,0]
    elif word_label == 'Pear': return [0,0,0,0,0,0,1,0,0,0]
    elif word_label == 'Pineapple': return [0,0,0,0,0,0,0,1,0,0]
    elif word_label == 'Strawberries': return [0,0,0,0,0,0,0,0,1,0]
    elif word_label == 'Watermelon': return [0,0,0,0,0,0,0,0,0,1]

# Preprocess the training and test data
def create_data(dir):
    data = []
    labels = []
    for img in tqdm(os.listdir(dir)):
        label = label_img(img)
        path = os.path.join(dir,img)
        img = cv2.imread(path)
        g = np.resize(img, (32, 32, 3))
        data.append([np.array(g)])
        labels.append([np.array(label)])
    return data, labels


# Initialize and format the data into arrays
train_x, train_y = create_data(TRAIN_DIR)
test_x, test_y = create_data(TEST_DIR)
train_x = np.reshape(train_x, (12680, 32, 32, 3))
train_x = train_x.astype(np.float32)
train_y = np.reshape(train_y, (12680, 10))
test_x = np.reshape(test_x, (1200, 32, 32, 3))
test_x = test_x.astype(np.float32)
test_y = np.reshape(test_y, (1200, 10))

X, Y, X_test, Y_test = train_x, train_y, test_x, test_y

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
network = conv_2d(network, 64, 3, activation = 'relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)


model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=10, run_id='fruits_cnn')


