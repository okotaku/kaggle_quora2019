import time
import torch
import numpy as np
from tqdm import tqdm

from src.logger import LOGGER
from src.utils import sigmoid, threshold_search


class Trainer:
    def __init__(self, model, train_loader, valid_loader, y_val, test_loader, loss, optimizer, scheduler,
                 model_path, epochs=4, batch_size=512, clip=0.5):

        self.model = model.cuda()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.y_val = y_val
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_path = model_path
        self.loss = loss
        self.batch_size = batch_size
        self.clip = clip

    def run_train_epoch(self):
        self.model.train()
        avg_loss = 0.

        for x_batch, y_batch in tqdm(self.train_loader, disable=True):
            y_pred = self.model(x_batch)
            loss_ = self.loss(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss_.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
            self.optimizer.step()

            avg_loss += loss_.item() / len(self.train_loader)

        return avg_loss

    def run_validation_epoch(self):
        self.model.eval()
        avg_val_loss = 0.
        valid_preds = np.zeros((len(self.valid_loader.dataset)))

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(self.valid_loader):
                y_pred = self.model(x_batch).detach()
                avg_val_loss += self.loss(y_pred, y_batch).item() / len(self.valid_loader)
                valid_preds[i * self.batch_size:(i + 1) * self.batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
        search_result = threshold_search(self.y_val.cpu().numpy(), valid_preds)

        val_f1, val_threshold = search_result['f1'], search_result['threshold']

        return valid_preds, avg_val_loss, val_f1

    def run_test_epoch(self):
        self.model.eval()
        test_preds = np.zeros((len(self.test_loader.dataset)))

        with torch.no_grad():
            for i, (x_batch,) in enumerate(self.test_loader):
                y_pred = self.model(x_batch).detach()
                test_preds[i * self.batch_size:(i + 1) * self.batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        return test_preds

    def save_model(self):
        with open(self.model_path, "wb") as fout:
            torch.save(self.model.state_dict(), fout)
        LOGGER.info("Model was saved in {}".format(self.model_path))

    def load_model(self):
        with open(self.model_path, "rb") as fin:
            state_dict = torch.load(fin)

        self.model.load_state_dict(state_dict)

    def run(self, validate=False):
        best_f1 = 0
        for epoch in range(self.epochs):
            start_time = time.time()

            avg_loss = self.run_train_epoch()

            if validate:
                valid_preds, avg_val_loss, val_f1 = self.run_validation_epoch()
                elapsed_time = time.time() - start_time
                LOGGER.info('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_f1={:.4f}  \t time={:.2f}s'.format(
                        epoch + 1, self.epochs, avg_loss, avg_val_loss, val_f1, elapsed_time))
                if val_f1 > best_f1:
                    self.save_model()
                    best_f1 = val_f1
            else:
                elapsed_time = time.time() - start_time
                LOGGER.info('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, self.epochs, avg_loss, elapsed_time))
                self.save_model()
            self.scheduler.step()

        if not validate or val_f1 != best_f1:
            self.load_model()
            valid_preds, avg_val_loss, val_f1 = self.run_validation_epoch()
            elapsed_time = time.time() - start_time
            LOGGER.info('Fin {}epochs \t loss={:.4f} \t val_loss={:.4f} \t val_f1={:.4f}  \t time={:.2f}s'.format(
                self.epochs, avg_loss, avg_val_loss, val_f1, elapsed_time))

        test_preds = self.run_test_epoch()

        return valid_preds, test_preds
