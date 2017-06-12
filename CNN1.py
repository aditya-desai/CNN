# Import libraries and modules
from timeit import default_timer as timer
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      
import tensorflow as tf
from PIL import Image, ImageFilter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start = timer()

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
        g = np.resize(img, (1, 32, 32, 3))
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

n_classes = 10                                # Number of output categories     
batch_size = 128                             # Number of training examples to go through in one iteration  
hm_epochs = 10                          # Number of times to run through the entire training data
lr = 0.005

x = tf.placeholder('float', [None, 32, 32, 3])
y = tf.placeholder('float')

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5, 5, 3, 32])),
               'W_conv2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc':tf.Variable(tf.random_normal([8*8*64, 1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, [-1, 32, 32, 3])

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

#wrong = []
#def sort_wrong_images():
#    predy = []
#    for row in test_y:
#        predy.append(np.argmax(row))
#    predy = np.array(predy)
#    wrong = predi - predy
#    accrcheck = wrong
#    accrcheck = np.array(accrcheck)
#    for _ in range(3000):
#       if accrcheck[_] != 0:
#            accrcheck[_] = 1
#    wrong_imageind = []
#    for row in range(3000):
#        if wrong[row] != 0:
#            wrong_imageind.append(row)
#    list(wrong_imageind)
#    return _, accrcheck, predy, row, wrong, wrong_imageind

#_, accrcheck, predy, row, wrong, wrong_imageind = sort_wrong_images()

#end = timer()

#def Write_to_file():
#    file = open('results.txt', 'a')
#    file.write("\n\n\nLog\n")
#    file.write("\nNumber of training images = " + str(len(train_y)) + "\n")
#    file.write("\nNumber of test images = " + str(len(test_y)) + "\n")
#    file.write("\nNumber of nodes in each layer =" + "[" + str(n_nodes_hl1) + ", " + str(n_nodes_hl2) + ", " + str(n_nodes_hl3) +"]\n")
#    file.write("\nNumber of epochs run = " + str(hm_epochs) + "\n")
#    file.write("\nLearning rate = " + str(lr) + "\n")
#    file.write("\nAccuracy achieved by the network = " + str(np.mean(1 - accrcheck)) + "\n")
#    file.write("\nTime taken for execution = " + str(end-start) + "\n")
#    file.write("\nThe list of indices of wrongly classified images in the test set : " + str(wrong_imageind))
#    file.close()
#    return file

#file = Write_to_file()



