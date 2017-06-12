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
        g = np.reshape(img, (1, 1024))
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
        g = np.reshape(img, (1, 1024))
        testing_data.append([np.array(g)])
        labels_test.append([np.array(label)])
    return testing_data, labels_test

# Initialize and format the data into arrays
train_x, train_y = create_train_data()
test_x, test_y = create_test_data()
train_x = np.reshape(train_x, (15300, 1024))
train_x = train_x.astype(np.float32)
train_y = np.reshape(train_y, (15300, 10))
test_x = np.reshape(test_x, (3000, 1024))
test_x = test_x.astype(np.float32)
test_y = np.reshape(test_y, (3000, 10))

# We'll take 3 hidden layers
n_nodes_hl1 = 1000                             # Number of nodes in each hidden layer
n_nodes_hl2 = 1000                                              
n_nodes_hl3 = 1000

n_classes = 10                                # Number of output categories     
batch_size = 100                              # Number of training examples to go through in one iteration  
hm_epochs = 10                          # Number of times to run through the entire training data
lr = 0.005

x = tf.placeholder('float', [None, 1024])
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([1024, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}     # define structure of different layers

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Model of the neural network : 1 input layer -> 3 hidden layers -> 1 output layer
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output


# Training and testing the network using the training data (Tensorflow library used for cost and optimizer functions)
def train_neural_network(x):
    prediction = neural_network_model(x)
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



