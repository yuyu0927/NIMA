import os
os.environ.setdefault('PATH', '')
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD
from keras import backend as K
import tensorflow as tf
import utils


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
for layer in base_model.layers:
    #print(layer)
    layer.trainable = False

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

train_X_raw, train_y_raw = utils.load_data()

train_X_raw = train_X_raw
train_y_raw = train_y_raw

train_X_raw, test_X_raw, train_y_raw, test_y_raw = train_test_split(train_X_raw, train_y_raw, test_size=0.1, random_state=42)

test_X = np.zeros([0, image_size, image_size, 3])
for image in range(test_X_raw.shape[0]):
    pic = test_X_raw[image]
    img = resize(pic,(image_size,image_size,3))
    temp = np.reshape(img, (1, image_size,image_size,3))
    test_X = np.append(test_X, temp, axis=0)
test_y = test_y_raw
print('test size:')
print(test_X.shape)

train_X = np.zeros([0, image_size, image_size, 3])
for image in range(train_X_raw.shape[0]):
    pic = train_X_raw[image]
    img = resize(pic,(image_size,image_size,3))
    temp = np.reshape(img, (image_size,image_size,3))
    train_X = np.append(train_X, np.expand_dims(temp, axis=0), axis=0)
train_y = train_y_raw

print()
print(train_X.shape)

train_X_1 = train_X_raw[:, 0:image_size, 0:image_size, :]
train_y_1 = train_y_raw
train_X = np.concatenate((train_X_1, train_X), axis=0)
train_y = np.concatenate((train_y_1, train_y), axis=0)

train_X_2 = train_X_raw[:, 0:image_size, -image_size:, :]
train_y_2 = train_y_raw
train_X = np.concatenate((train_X_2, train_X), axis=0)
train_y = np.concatenate((train_y_2, train_y), axis=0)

train_X_3 = train_X_raw[:, -image_size:, 0:image_size, :]
train_y_3 = train_y_raw
train_X = np.concatenate((train_X_3, train_X), axis=0)
train_y = np.concatenate((train_y_3, train_y), axis=0)

train_X_4 = train_X_raw[:, -image_size:, -image_size:, :]
train_y_4 = train_y_raw
train_X = np.concatenate((train_X_4, train_X), axis=0)
train_y = np.concatenate((train_y_4, train_y), axis=0)

flip_train_X = train_X[:, :, ::-1, :]
flip_train_y = train_y[:,:]
train_X = np.concatenate((flip_train_X, train_X), axis=0)
train_y = np.concatenate((flip_train_y, train_y), axis=0)
print()
print(train_X.shape)
print()

sess = tf.Session()
n_epochs = 20
batch_size = 50
n_batches = train_X.shape[0] // batch_size
lr = 0.003
init = tf.global_variables_initializer()
sess.run(init)

valid_best = 1000
for epoch in range(n_epochs):
    train_cost = 0
    train_X, train_y = shuffle(train_X, train_y, random_state=43)
    if 1:
        if epoch%10 == 0:
            lr = lr/2
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        _, _cost = sess.run([train, cost], feed_dict={x_input: train_X[start: end], t: train_y[start: end], learning_rate:lr})
        train_cost += _cost
    valid_cost, pred_y = sess.run([cost, y], feed_dict={x_input: test_X, t: test_y})
    if valid_cost < valid_best:
        model.save('weights/vgg_tf.h5')
        valid_best = valid_cost
    print('EPOCH:{0:d}, train_cost:{1:.5f}, valid_cost:{2:.5f}' .format(epoch + 1, train_cost/train_X.shape[0], valid_cost/test_X.shape[0]))