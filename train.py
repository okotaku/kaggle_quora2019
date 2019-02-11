import os
import gc
import torch
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences
from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings

warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 due to no predicted samples.")

from src.model import NeuralNet
from src.trainer import Trainer
from src.logger import setup_logger
from src.preprocessing import preprocess
from src.get_embedding import GetEmbedding
from src.utils import df_parallelize_run, threshold_search, seed_everything, mkdir


def main(train_path, test_path, max_features, max_len, glove_path, para_path, model_save_path,
         epochs=4, batch_size=512, seed=1029):
    logger_path = os.path.join(model_save_path, "log.txt")
    setup_logger(out_file=logger_path)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = df_parallelize_run(train, text_clean_wrapper)
    test = df_parallelize_run(test, text_clean_wrapper)

    tk = Tokenizer(lower=True, filters='', num_words=max_features)
    full_text = list(train['question_text'].values) + list(test['question_text'].values)
    tk.fit_on_texts(full_text)
    train_tokenized = tk.texts_to_sequences(train['question_text'].fillna('missing'))
    test_tokenized = tk.texts_to_sequences(test['question_text'].fillna('missing'))
    word_index = tk.word_index

    X_train = pad_sequences(train_tokenized, maxlen=max_len)
    X_test = pad_sequences(test_tokenized, maxlen=max_len)

    y_train = train['target'].values

    get_embedding = GetEmbedding(max_features, word_index)
    glove = get_embedding.load(glove_path, emb_mean=-0.005838499, emb_std=0.48782197)
    para = get_embedding.load(para_path, emb_mean=-0.0053247833, emb_std=0.49346462)
    embedding_matrix = glove * 0.8 + para * 0.2
    del glove, para;
    gc.collect()

    x_test_cuda = torch.tensor(X_test, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=10).split(X_train, y_train))

    seed_everything()

    train_preds = np.zeros(len(train))
    test_preds = np.zeros((len(test), len(splits)))

    for i, (train_idx, valid_idx) in enumerate(splits):
        model_path = os.path.join(model_save_path, "model_fold{}".format(i + 1))
        x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {i + 1}')

        seed_everything(seed + i)
        model = NeuralNet(max_features, max_len, embedding_matrix)
        model.cuda()

        optimizer = torch.optim.Adam(model.parameters())
        scheduler = CosineAnnealingLR(optimizer, T_max=3)
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()

        trainer = Trainer(model, train_loader, valid_loader, y_val_fold, test_loader, loss, optimizer, scheduler,
                          model_path, epochs, batch_size)
        valid_preds_fold, test_preds_fold = trainer.run(validate=True)

        train_preds[valid_idx] = valid_preds_fold
        test_preds[:, i] = test_preds_fold

    search_result = threshold_search(y_train, train_preds)
    print(search_result)
    test_preds = test_preds.mean(1) > search_result['threshold']
    return test_preds


def text_clean_wrapper(df):
    df["question_text"] = df["question_text"].apply(preprocess)
    return df


if __name__ == "__main__":
    max_features = 120000
    max_len = 72
    train_path = "../input/train.csv"
    test_path = "../input/test.csv"
    glove_path = '../input/glove.840B.300d/glove.840B.300d.txt'
    para_path = '../input/paragram_300_sl999/paragram_300_sl999.txt'
    model_save_path = "./model/"
    mkdir(model_save_path)

    test_preds = main(train_path, test_path, max_features, max_len, glove_path, para_path, model_save_path)

    sub = pd.read_csv('../input/sample_submission.csv')
    sub['prediction'] = test_preds
    sub.to_csv("submission.csv", index=False)