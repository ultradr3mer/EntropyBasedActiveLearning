import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from deepLearner import DeepLearner, model_file_name
from generator import Generator
from learnerDataset import LearnerDataset
import plotFunctions


class DeepTrainer(object):
    def __init__(self):
        self.gen = Generator()
        self.data_set = self.gen.load_or_create_dataset()
        self.counter = 0
        self.random = random.Random('sidjhf')
        self.learner = DeepLearner()

    def train_step(self, dataloader, model, loss_fn, optimizer):
        device = self.learner.device
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, model, loss_fn):
        device = self.learner.device
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0  # , correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    def train(self):
        model = self.learner.model
        indexed_data = list(enumerate(self.data_set))
        train, test = train_test_split(indexed_data, test_size=0.1, shuffle=True)
        train_dataloader = self.setup_dataloader(train)
        test_dataloader = self.setup_dataloader(test)

        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        epochs = 50
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            scheduler.step(t)
            self.train_step(train_dataloader, model, loss_fn, optimizer)
            self.test(test_dataloader, model, loss_fn)
        print("Done!")

        torch.save(model.state_dict(), model_file_name)

        def create_image(d):
            r = self.gen.normalize(d[0])
            g = self.gen.normalize(d[1])
            b = self.gen.normalize(d[2])
            return np.dstack((r,g,b))

        i_pred = 0
        i_actual = 0
        i_x = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(self.learner.device), y
                pred = model(X)
                for p in pred:
                    self.gen.save_as_image(self.gen.normalize(p.cpu().numpy()[0]), f'result/{i_pred}_pred.png')
                    i_pred += 1
                for item in y:
                    self.gen.save_as_image(self.gen.normalize(item.cpu().numpy()[0]), f'result/{i_actual}_actual.png')
                    i_actual += 1
                for item in X:
                    self.gen.save_as_image(create_image(item.cpu().numpy()), f'result/{i_x}_x.png')
                    i_x += 1

    def setup_dataloader(self, data):
        data_full = list([self.gen.load_or_gen_train_data(item, i) for i, item in data])
        my_dataset = LearnerDataset(data_full)
        return DataLoader(my_dataset, batch_size=32, shuffle=True)


if __name__ == '__main__':
    trainer = DeepTrainer()
    trainer.train()
