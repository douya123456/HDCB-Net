from keras.models import Model
from keras.layers import Input,concatenate
from keras.layers.convolutional import Convolution2D, UpSampling2D,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization


IMAGE_W = 224
IMAGE_H = 224
#jwb
IMAGE_C = 3



def dailt_conv(income,rate=1,filter_num=16,wei_init='uniform',activation=LeakyReLU(),name='dailt_conv'):

    conv = Convolution2D(filter_num, kernel_size=(1, 1), strides=(1, 1), padding="same",kernel_initializer=wei_init,activation=None)(income)
    bn = BatchNormalization()(conv)
    act = activation(bn)

    conv = Convolution2D(filter_num, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=wei_init, activation=None)(act)
    act = BatchNormalization()(conv)

    dconv = Convolution2D(filter_num, kernel_size=(3, 3), strides=(1, 1), padding="same",kernel_initializer=wei_init,dilation_rate=(rate,rate),activation=None)(act)
    bn = BatchNormalization()(dconv)

    conv = Convolution2D(filter_num, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer=wei_init,activation=None)(bn)
    bn = BatchNormalization()(conv)
    act = activation(bn)

    return act


def CrackSuffleNet(
        input_shape=(IMAGE_W,IMAGE_H,IMAGE_C),
        n_labels=2,
        output_mode="sigmoid"):

    filter_num = 64
    wei_init = 'he_normal'
    act = LeakyReLU()


    # encoder
    inputs = Input(shape=input_shape)

    c0 = dailt_conv(inputs, rate=1, filter_num=filter_num, wei_init=wei_init, activation=act, name='conv1')
    p0 = MaxPooling2D(strides=2)(c0)

    c1 = dailt_conv(p0, rate=1, filter_num=filter_num * 2, wei_init=wei_init, activation=act, name='conv2')
    p1 = MaxPooling2D(strides=2)(c1)

    c2 = dailt_conv(p1, rate=1, filter_num=filter_num * 4, wei_init=wei_init, activation=act, name='conv3')
    p2 = MaxPooling2D(strides=2)(c2)
    p2 = Dropout(rate=0.3)(p2)

    c4 = dailt_conv(p2, rate=2, filter_num=filter_num * 4, wei_init=wei_init, activation=act, name='dailt_conv1')
    c4 = dailt_conv(c4, rate=3, filter_num=filter_num * 4, wei_init=wei_init, activation=act, name='dailt_conv2')
    c4 = dailt_conv(c4, rate=5, filter_num=filter_num * 4, wei_init=wei_init, activation=act, name='dailt_conv3')

    c5 = dailt_conv(c4, rate=2, filter_num=filter_num * 4, wei_init=wei_init, activation=act, name='dailt_conv4')
    c5 = dailt_conv(c5, rate=5, filter_num=filter_num * 4, wei_init=wei_init, activation=act, name='dailt_conv5')
    c5 = dailt_conv(c5, rate=9, filter_num=filter_num * 4, wei_init=wei_init, activation=act, name='dailt_conv6')
    p3 = dailt_conv(c5, rate=13, filter_num=filter_num * 4, wei_init=wei_init, activation=act, name='dailt_conv9')

    # decoder
    unpool_1 = concatenate([UpSampling2D(size=(2, 2),interpolation='bilinear')(p3), c2], axis=3)
    c4 = dailt_conv(unpool_1, rate=1, filter_num=filter_num * 4, wei_init=wei_init, activation=act, name='conv4')

    unpool_2 = concatenate([UpSampling2D(size=(2, 2),interpolation='bilinear')(c4), c1], axis=3)
    c5 = dailt_conv(unpool_2, rate=1, filter_num=filter_num * 2, wei_init=wei_init, activation=act, name='conv5')

    unpool_3 = concatenate([UpSampling2D(size=(2, 2),interpolation='bilinear')(c5), c0], axis=3)
    c6 = dailt_conv(unpool_3, rate=1, filter_num=filter_num, wei_init=wei_init, activation=act, name='conv6')

    c8 = Convolution2D(n_labels, (1, 1),strides=(1, 1), padding="same",kernel_initializer=wei_init,activation=None)(c6)
    outputs = Convolution2D(1, (1, 1), strides=(1, 1), padding="same", kernel_initializer=wei_init, activation=output_mode)(c8)

    model = Model(inputs=inputs, outputs=outputs, name="CrackSuffleNet")

    print(model.summary())

    return model


# model = CrackSuffleNet()