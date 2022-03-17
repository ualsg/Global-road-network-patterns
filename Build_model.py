#!/usr/bin/env python

from __future__ import print_function
from tensorflow import keras
from MODEL import MODEL,ResnetBuilder
import sys
sys.setrecursionlimit(10000)

# from keras import backend as K

class Build_model(object):
    def __init__(self,config):
        #self.checkpoints = config.checkpoints
        self.normal_size = config.normal_size
        self.channles = config.channles
        self.classNumber = config.classNumber
        self.model_name = config.model_name
        self.config = config


    def model_compile(self,model):
        #adam = keras.optimizers.Adam(lr=self.lr)
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"])  # compile之后才会更新权重和模型
        return model

    def build_model(self):
        model = ResnetBuilder().build_resnet34(self.config)
        model = self.model_compile(model)
        return model

    def build_mymodel(self):
        model = ResnetBuilder().build_myresnet(self.config)
        model = self.model_compile(model)
        return model