import os
import torch
import random
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics import f1_score


def df_parallelize_run(df, func):
    num_partitions = 20
    num_cores = psutil.cpu_count()
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()

    return df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass
