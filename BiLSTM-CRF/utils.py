import logging
import numpy as np
from sklearn.model_selection import train_test_split
import config
def set_logger(log_path):

    logger = logging.getLogger()#用logging.getLogger(name)方法进行初始化
    logger.setLevel(logging.INFO)#设置级别

    if not logger.handlers:#如果没有logger.handlers:添加
        # Logging to a file
        file_handler = logging.FileHandler(log_path)#地址
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def load_data(dataset_dir):
    data = np.load(dataset_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    return word_test,label_test
