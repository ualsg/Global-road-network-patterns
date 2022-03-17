# -*- coding: utf-8 -*-

import os
import random
import sys
class DefaultConfig():

    try:
        model_name = sys.argv[1]
    except:
        print("use default model ResNet34, see config.py")
        model_name = "ResNet34-6class-aug2"
    with open('train_class_idx.txt', 'r') as f:
        classes = f.read().split('\n')
        classNumber = len(classes)
        #f.write('\n'.join(label_imgs.keys()))
    normal_size = 224
    channles = 3  # or 3 or 1 or 6

    is_mono = False



config = DefaultConfig()