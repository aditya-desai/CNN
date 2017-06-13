# Import libraries and modules
from timeit import default_timer as timer
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      
import tensorflow as tf

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
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        g = np.resize(img, (32, 32, 1))
        data.append([np.array(g)])
        labels.append([np.array(label)])
    return data, labels


# Initialize and format the data into arrays
train_x, train_y = create_data(TRAIN_DIR)
test_x, test_y = create_data(TEST_DIR)
train_x = np.reshape(train_x, (12680, 32, 32, 1))
train_x = train_x.astype(np.float32)
train_y = np.reshape(train_y, (12680, 10))
test_x = np.reshape(test_x, (1200, 32, 32, 1))
test_x = test_x.astype(np.float32)
test_y = np.reshape(test_y, (1200, 10))

n_classes = 10                                  # Number of output categories     
batch_size = 128                                # Number of training examples to go through in one iteration  
hm_epochs = 10                                 # Number of times to run through the entire training data
lr = 0.005                                      # Set the learning rate for the optimizer    

x = tf.placeholder('float', [None, 32, 32, 1])
y = tf.placeholder('float')
keep_prob = tf.placeholder(tf.float32)          # Probability that we keep a neuron during backprop (used in dropout fn)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def conv2d(x, w):                                                       # Define the convolution operation
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def maxpool2d(x):                                                       # Define the pooling operation
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5, 5, 1, 32])),      # Dictionary describing the weight variables
               'W_conv2':tf.Variable(tf.random_normal([5, 5, 32, 64])),      
               'W_fc1':tf.Variable(tf.random_normal([8*8*64, 1024])),
               'W_fc2':tf.Variable(tf.random_normal([1024, 1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),                # Dictionary describing the bias variables
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc1':tf.Variable(tf.random_normal([1024])),
               'b_fc2':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, [-1, 32, 32, 1])

    conv1 = conv2d(x, weights['W_conv1'] + biases['b_conv1'])           # 1st conv layer
    conv1 = maxpool2d(conv1)                                            # 1st pooling layer
                                                                        
    conv2 = conv2d(conv1, weights['W_conv2'] + biases['b_conv2'])       # 2nd conv layer
    conv2 = maxpool2d(conv2)                                            # 2nd pooling layer

    fc1 = tf.reshape(conv2, [-1, 8*8*64])                               
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1']) + biases['b_fc1'])    # 1st fully connected layer
    
    fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2']) + biases['b_fc2'])    # 2nd fully connected layer
    
    fc2_dropout = tf.nn.dropout(fc2, keep_prob)                             # dropout to reduce overfitting
    
    output = tf.matmul(fc2_dropout, weights['out']) + biases['out']         # output layer

    return output



# Training and testing the network using the training data (Tensorflow library used for cost and optimizer functions)
def train_neural_network(x, train_data, train_labels):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        #saver.restore(sess, 'C:/Users/Admin/Documents/Visual Studio 2017/Projects/CNN1/trained.ckpt')
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        prediction = tf.argmax(prediction, 1)
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            train_data, train_labels = randomize(train_data, train_labels)
            epoch_loss = 0
            i=0
            while i < len(train_data):
                start = i
                end = i+batch_size
                batch_x = np.array(train_data[start:end])
                batch_y = np.array(train_labels[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
                epoch_loss += c
                i+=batch_size

            print('Epoch', epoch+1, 'completed out of', hm_epochs,'Current Epoch Loss:',epoch_loss)
            if epoch % 5 == 0:
                print('Training Accuracy:',accuracy.eval({x:train_data, y:train_labels, keep_prob: 1.0}))
                print('Testing Accuracy:',accuracy.eval({x:test_x, y:test_y, keep_prob: 1.0}))

        save_path = saver.save(sess, 'C:/Users/Admin/Documents/Visual Studio 2017/Projects/CNN1/trained.ckpt')
        print("Model saved in file: {}".format(save_path))
        print('Training Accuracy:',accuracy.eval({x:train_data, y:train_labels, keep_prob: 1.0}))
        print('Testing Accuracy:',accuracy.eval({x:test_x, y:test_y, keep_prob: 1.0}))

        return 
    
train_neural_network(x, train_x, train_y)

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




