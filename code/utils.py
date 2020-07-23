import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K


def dice_coefficient(y_true,y_pred):
    smooth = 1
    y_true = tf.to_float(y_true)
    y_pred = tf.to_float(y_pred)
    intersection = tf.reduce_sum(y_pred * y_true)+ smooth
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    loss = 1.- (2 * intersection / union)

    return loss

# Define Jaccard
def jaccard_coefficient(y_true,y_pred):

    with tf.name_scope('Accuracy'):
        smooth = 1e-4
        y_true = tf.to_float(y_true)
        y_pred = tf.to_float(y_pred)
        # y_true = tf.to_int32(y_true > 0.5)
        # y_pred = tf.to_int32(y_pred > 0.5)
        intersection = tf.reduce_sum(y_pred * y_true) + smooth
        union = tf.reduce_sum(y_pred) + smooth
        accuracy = (intersection / union)

    return accuracy



# tversky loss
def tversky(y_true, y_pred):
    smooth = 1e-4
    y_true = tf.to_float(y_true)
    y_pred = tf.to_float(y_pred)
    # y_true = tf.to_int32(y_true > 0.5)
    # y_pred = tf.to_int32(y_pred > 0.5)
    true_pos = tf.reduce_sum(y_true * y_pred)
    false_neg = tf.reduce_sum(y_true * (1-y_pred))
    false_pos = tf.reduce_sum((1-y_true)*y_pred)
    alpha = 0.75
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)


def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)



# alpha取值范围0~1，当alpha>0.5时，可以相对增加y=1所占的比例。实现正负样本的平衡。
def focal_loss(y_true,y_pred):
    gamma = 0.15
    alpha = 0.85
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


# 使用-log放大Dice Loss，使用alpha缩小Focal Loss
def mixedLoss(y_true,y_pred,alpha=0.08):
    return alpha * focal_loss(y_true,y_pred) - K.log(dice_coefficient(y_true,y_pred))


# Define IoU metric，评价指标,重叠度
def mean_iou_accuracy_op(y_true,y_pred, x):
    with tf.name_scope('Accuracy'):
        prec = []
        for t in np.arange(0.5, 1.0, 0.5):
            y_pred_tmp = tf.to_int32(y_pred > 0.5)#预测值大于0.5
            score, update_op = tf.metrics.mean_iou(y_true, y_pred_tmp, 2)#update_op才是真正负责更新变量，而第一个score只是简单根据当前变量计算评价指标，后一个作用是更新变量，另外会同时返回一个结果，对于tf.metric.accuracy，就是更新变量后实时计算的accuracy。
            with tf.Session() as sess:
                sess.run(tf.local_variables_initializer())#初始化变量
            with tf.control_dependencies([update_op]):#用来控制计算流图的，给图中的某些计算指定顺序
                score = tf.identity(score)
            prec.append(score)
        acc = tf.reduce_mean(tf.stack(prec), axis=0, name='mean_iou')#取平均值
    return acc


# focal loss with multi label
def multi_focal_lossl(classes_num, gamma=2., alpha=.25, e=0.3):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''

        #1# get focal loss with no balanced weight which presented in paper function (4)
        FT = -1 * K.pow((1. - prediction_tensor), gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-3, .999))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [total_num / ff for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(target_tensor, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed