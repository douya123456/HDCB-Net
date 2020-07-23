# test_model.py
import numpy as np
import cv2
import os
from model import CrackSuffleNet
import time

# GPU
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# 指定第一块GPU可用
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def load_model():
    model = CrackSuffleNet()
    model_path = 'checkpoint-dir/weights-0003-0.399410.hdf5'
    # Load a model
    model.load_weights(model_path)

    return model

def crop_stride(slide_h,slide_w):

    TEST_PATH = 'test/'
    test_ids = next(os.walk(TEST_PATH))[2]
    test_ids.sort()

    model = load_model()

    start_time = time.time()  # 记录时间

    crop = []
    for name in test_ids:
        path = TEST_PATH + name
        img = cv2.imread(path,1)
        img = img.astype(np.float32) / 255.0

        slide_w = slide_w
        slide_h = slide_h

        size_w = 224
        size_h = 224

        img_w = img.shape[1]
        img_h = img.shape[0]


        # finalI用来融合小图
        finalI = np.zeros((img_h, img_w), dtype=np.float32)
        # 这里python自我防溢出
        for w in range(0,img_w,slide_w):
            for h in range(0,img_h,slide_h):
                    block = img[h:h+size_h,w:w+size_w]

                    temp_h = block.shape[0]
                    temp_w = block.shape[1]
                    tempI = np.zeros((size_h, size_w, 3), dtype=np.float32)
                    tempI[0:temp_h,0:temp_w] = block

                    tempI = np.expand_dims(tempI, axis=0)
                    y_pred = model.predict(tempI)
                    y_pred = (y_pred > 0.35).astype(np.uint8)
                    y_pred = np.squeeze(y_pred)
                    finalBlock = y_pred[0:temp_h,0:temp_w]

                    finalI[h:h + size_h, w:w + size_w] = finalBlock
                    crop.append(finalBlock)

        label_ = finalI.astype(np.float32) * 255
        path = name
        cv2.imwrite(path, label_)

    duration = time.time() - start_time  # 时间
    print('testing time total %.3f sec\n' % (duration))
    print('testing each picture %.3f sec\n' % (duration / len(test_ids)))
    print('testing each block %.3f sec\n' % (duration / len(crop)))
    print("finish")

# 以一定步长划窗测试大图
crop_stride(180,180)