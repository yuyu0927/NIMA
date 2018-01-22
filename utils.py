import os
os.environ.setdefault('PATH', '')
import numpy as np
import glob
from skimage import io
from skimage.transform import resize

# path to the images and the text file which holds the scores and ids
base_images_path = 'distorted_images/'
dataset_path = 'class.txt'

IMAGE_SIZE = 299

files = glob.glob(base_images_path + "*.bmp")
files = sorted(files)

train_image_paths = []
train_scores = []

print("Loading training set and val set")
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

def load_data():
    imgs = np.zeros([0, 299, 299, 3])
    labels = []
    for i in range(3000):
        img_tmp = io.imread(train_image_paths[i])
        imgs = np.append(imgs, np.expand_dims(resize(img_tmp,(IMAGE_SIZE,IMAGE_SIZE,3)), axis=0), axis=0)
        labels.append(train_scores[i])
        print(imgs.shape)
        if i == 300:
            break
    label_y = np.array(labels)

    print(imgs.shape)
    print(label_y.shape)
    return imgs,label_y



