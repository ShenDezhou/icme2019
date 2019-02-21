#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2019年2月20日

@author: BD-PC50
'''
import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from keras.callbacks import Callback
from keras.backend import backend

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

