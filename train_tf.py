import os
os.environ.setdefault('PATH', '')
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout
#from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD
from keras import backend as K
import tensorflow as tf
#import utils
K.set_learning_phase(1)

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

image_size = 224

base_model = VGG16(include_top=True, pooling='avg')
base_model.layers.pop()
base_model_output =base_model.layers[-1].output
base_model.layers[-1].outbound_nodes = []
# for layer in base_model.layers:
#     #print(layer)
#     layer.trainable = False

x = Dropout(0.75)(base_model_output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.summary()


x_input = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = model(x_input)
t = tf.placeholder(tf.float32, [None, 10])
learning_rate = tf.placeholder(tf.float32, [])
cost = earth_mover_loss(t, y)
train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)
######################################################################################
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.transform import resize

base_images_path = 'distorted_images/'
dataset_path = 'class.txt'
train_image_paths = []
train_scores = []
with open(dataset_path, mode='r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.split()
       
        values = np.array(token[:10], dtype='float32')

        file_path = base_images_path + token[10]
        if os.path.exists(file_path):
            train_image_paths.append(file_path)
            train_scores.append(values)
        elif os.path.exists(base_images_path + 'I'+token[10][1:]):
            train_image_paths.append(base_images_path + 'I'+token[10][1:])
            train_scores.append(values)
        elif os.path.exists(base_images_path + 'I'+token[10][1:-3]+'BMP'):
            train_image_paths.append(base_images_path + 'I'+token[10][1:-3]+'BMP')
            train_scores.append(values)

        count = 3000 // 20
        if i % count == 0 and i != 0:
            print('Loaded %d percent of the dataset' % (i / 3000. * 100))

train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')
train_image_paths, val_image_paths,train_scores,val_scores= train_test_split(train_image_paths, train_scores, test_size=0.1, random_state=42)
IMAGE_SIZE = 224
batch_size = 100
def parse_data(filename, scores):
    '''
    Loads the image file, and randomly applies crops and flips to each image.
    Args:
        filename: the filename from the record
        scores: the scores from the record
    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize_images(image, (256, 256))
    image = tf.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.image.random_flip_left_right(image)
    return image, scores

def parse_data_without_augmentation(filename, scores):
    '''
    Loads the image file without any augmentation. Used for validation set.
    Args:
        filename: the filename from the record
        scores: the scores from the record
    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image, scores

sess_data = tf.Session()
train_dataset = tf.data.Dataset().from_tensor_slices((train_image_paths, train_scores))
train_dataset = train_dataset.map(parse_data, num_parallel_calls=2)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()
if shuffle:
    train_dataset = train_dataset.shuffle(buffer_size=4)
train_iterator = train_dataset.make_initializable_iterator()
train_batch = train_iterator.get_next()
sess_data.run(train_iterator.initializer)

val_dataset = tf.data.Dataset().from_tensor_slices((val_image_paths, val_scores))
val_dataset = val_dataset.map(parse_data_without_augmentation)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.repeat()
val_iterator = val_dataset.make_initializable_iterator()
val_batch = val_iterator.get_next()
sess_data.run(val_iterator.initializer)


##########################################################################################
sess = tf.Session()
n_epochs = 40
n_batches = train_image_paths.shape[0] // batch_size
lr = 0.00003
init = tf.global_variables_initializer()
sess.run(init)

valid_best = 1000
for epoch in range(n_epochs):
    train_cost = 0
    valid_cost = 0
    if 1:
        if epoch%10 == 0:
            lr = lr/2
    for i in range(n_batches):
        X_batch, y_batch = sess_data.run(train_batch)
        _, _cost = sess.run([train, cost], feed_dict={x_input: X_batch, t: y_batch, learning_rate:lr})
        train_cost += _cost
    X_val_batch, y_val_batch = sess_data.run(val_batch)
    for i in range(val_image_paths.shape[0] // batch_size):
        _valid_cost, pred_y = sess.run([cost, y], feed_dict={x_input: X_val_batch, t: y_val_batch})
        valid_cost += _valid_cost
    if valid_cost < valid_best:
        model.save('weights/vgg_tf.h5')
        valid_best = valid_cost
    print('EPOCH:{0:d}, train_cost:{1:.5f}, valid_cost:{2:.5f}' .format(epoch + 1, train_cost , valid_cost))
sess_data.close()
