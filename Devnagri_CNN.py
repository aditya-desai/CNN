# Import libraries and modules
from timeit import default_timer as timer
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start = timer()

TRAIN_DIR = 'D:/Data/Devnagri/Train'
TEST_DIR =  'D:/Data/Devnagri/Test'

# Assign labels and convert them into one-hot arrays
def label_img(img):
    word_label = img.split(' ')[-2]
    if word_label == 'zero': return [1,0,0,0,0,0,0,0,0,0]
    elif word_label == 'one': return [0,1,0,0,0,0,0,0,0,0]
    elif word_label == 'two': return [0,0,1,0,0,0,0,0,0,0]
    elif word_label == 'three': return [0,0,0,1,0,0,0,0,0,0]
    elif word_label == 'four': return [0,0,0,0,1,0,0,0,0,0]
    elif word_label == 'five': return [0,0,0,0,0,1,0,0,0,0]
    elif word_label == 'six': return [0,0,0,0,0,0,1,0,0,0]
    elif word_label == 'seven': return [0,0,0,0,0,0,0,1,0,0]
    elif word_label == 'eight': return [0,0,0,0,0,0,0,0,1,0]
    elif word_label == 'nine': return [0,0,0,0,0,0,0,0,0,1]

# Preprocess the training data
def create_train_data():
    training_data = []
    labels_train = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        g = np.reshape(img, (1, 32, 32, 1))
        training_data.append([np.array(g)])
        labels_train.append([np.array(label)])
    return training_data, labels_train

# Preprocess the test data 
def create_test_data():
    testing_data = []
    labels_test = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        g = np.reshape(img, (1, 32, 32, 1))
        testing_data.append([np.array(g)])
        labels_test.append([np.array(label)])
    return testing_data, labels_test

# Initialize and format the data into arrays
train_x, train_y = create_train_data()
test_x, test_y = create_test_data()
train_x = np.reshape(train_x, (15300, 32, 32, 1))
train_x = train_x.astype(np.float32)
train_y = np.reshape(train_y, (15300, 10))
test_x = np.reshape(test_x, (3000, 32, 32, 1))
test_x = test_x.astype(np.float32)
test_y = np.reshape(test_y, (3000, 10))


n_classes = 10                                # Number of output categories     
batch_size = 100                              # Number of training examples to go through in one iteration  
hm_epochs = 15                          # Number of times to run through the entire training data
lr = 0.001

x = tf.placeholder('float', [None, 32, 32, 1])
y = tf.placeholder('float')

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc':tf.Variable(tf.random_normal([8*8*64, 1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, [-1, 32, 32, 1])

    conv1 = conv2d(x, weights['W_conv1'] + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['W_conv2'] + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 8*8*64])
    print(fc.get_shape())
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


# Training and testing the network using the training data (Tensorflow library used for cost and optimizer functions)
def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #saver.restore(sess, 'C:/Users/Admin/Documents/Visual Studio 2017/Projects/Devanagri2/trained.ckpt')
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size

            print('Epoch', epoch+1, 'completed out of', hm_epochs,'. Current Epoch Loss:',epoch_loss)
        save_path = saver.save(sess, 'C:/Users/Admin/Documents/Visual Studio 2017/Projects/Devanagri2/trained.ckpt')
        print("Model saved in file: {}".format(save_path))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        prediction = tf.argmax(prediction, 1)
        predi = sess.run(prediction, {x : test_x})
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
        return predi
    
predi = train_neural_network(x)
