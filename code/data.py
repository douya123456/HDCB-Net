# coding=utf-8
import matplotlib
matplotlib.use("Agg")
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
import os


seed = 7
np.random.seed(seed)

# data_shape = 448*448
img_w = 224
img_h = 224

DATA_PATH = '../augmentation'
TRAIN_PATH = DATA_PATH + '/train/image_positive/'
LABEL_PATH = DATA_PATH + '/train/label_positive/'
VAL_IMG_PATH = DATA_PATH + '/val/image_positive/'
VAL_LAB_PATH = DATA_PATH + '/val/label_positive/'



def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_w, img_h))
        img = np.array(img, dtype="float") / 255.0
        img[img > 0.37] = 1
        img[img <= 0.37] = 0
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(path,1)
        img = cv2.resize(img, (img_w, img_h))
        img = np.array(img, dtype="float") / 255.0
    return img

def get_train_val_name(train_path=TRAIN_PATH,image_val_path=VAL_IMG_PATH):

    train_set = next(os.walk(train_path))[2]
    train_set.sort()
    print(train_set,len(train_set))
    val_set = next(os.walk(image_val_path))[2]
    val_set.sort()
    print(val_set,len(val_set))

    return train_set, val_set


# data for training
def generateData(batch_size, data):
    # print('generateData...')
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(TRAIN_PATH + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(LABEL_PATH + url, grayscale=True)
            # print label.shape
            train_label.append(label)

            if batch % batch_size == 0:
                # print('get enough bacth!\n')
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0

# data for validation
def generateValidData(batch_size, data):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(VAL_IMG_PATH + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(VAL_LAB_PATH + url, grayscale=True)
            # print label.shape
            valid_label.append(label)

            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0
