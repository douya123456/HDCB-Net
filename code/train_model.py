from keras.callbacks import ModelCheckpoint
from data import get_train_val_name,generateData,generateValidData
from matplotlib import pyplot as plt
import numpy as np
import keras
from utils import tversky_loss,jaccard_coefficient,dice_coefficient

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


def train():
    EPOCHS = 500
    train_BS = 8
    val_BS = train_BS

    from model import CrackSuffleNet
    model = CrackSuffleNet()
    # model_path = 'checkpoint-dir/weights-0018-0.400025.hdf5'
    # Load a model
    # model.load_weights(model_path)

    adam = keras.optimizers.adam(lr=0.008)
    model.compile(loss=dice_coefficient, optimizer=adam, metrics=[jaccard_coefficient])
    print(model.summary())

    filepath = "checkpoint-dir/weights-{epoch:04d}-{val_loss:.6f}.hdf5"
    modelcheck = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True,mode='min',verbose=1)
    callable = [modelcheck]

    train_set,val_set = get_train_val_name()
    import random
    random.shuffle(train_set)
    random.shuffle(val_set)
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print ("the number of train data is",train_numb)
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(train_BS,train_set),steps_per_epoch=train_numb//train_BS,epochs=EPOCHS,verbose=1,
                    validation_data=generateValidData(val_BS,val_set),validation_steps=valid_numb//val_BS,callbacks=callable,max_q_size=1)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    fig_filepath = 'train_fig'
    plt.savefig(fig_filepath)


train()
