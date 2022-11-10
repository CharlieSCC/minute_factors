# -*- coding: utf-8 -*-
'''
@Project : min
@File : log_module.py
@Author : Shang Chencheng
@Date : 2022/8/22 13:11
'''
import os
import logging


def get_module_logger(log_path="./", log_file="xx1.log"):

    if not os.path.exists(log_path):
        # print('create path: %s' % model_path)
        os.makedirs(log_path)

    logger = logging.getLogger("info")
    logger.setLevel(logging.INFO)  # log level

    # create a handler

    fh = logging.FileHandler(log_path+log_file, mode='w')
    fh.setLevel(logging.INFO)  # log level of file
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # define print format
    fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add log to handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger