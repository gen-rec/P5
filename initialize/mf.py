import copy
import json
import os
import pickle
import random
from collections import defaultdict
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn.functional import normalize
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


class MF(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)

        nn.init.xavier_uniform_(self.user_factors.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.item_factors.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, user, item):
        user_factors = self.user_factors(user)
        item_factors = self.item_factors(item)
        return (user_factors * item_factors).sum(dim=1)


# noinspection PyTypeChecker
class MFTrainer:
    def __init__(
        self,
        device: torch.device,
        num_factors: int,
        data_path: str,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
    ):
        # Load Amazon ID to integer ID mapping
        datamaps = json.load(open(os.path.join(data_path, "datamaps.json"), "r", encoding="utf-8"))
        self.user2id = datamaps["user2id"]
        self.item2id = datamaps["item2id"]
        self.id2user = datamaps["id2user"]
        self.id2item = datamaps["id2item"]

        self.num_users = len(self.user2id)
        self.num_items = len(self.item2id)
        self.num_factors = num_factors

        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        # Load data
        data_splits = pickle.load(open(os.path.join(data_path, "exp_splits.pkl"), "rb"))
        raw_train = data_splits["train"]
        raw_val = data_splits["val"]
        raw_test = data_splits["test"]

        # Normalize ratings per user
        user_ratings = defaultdict(list)
        for data in raw_train:
            user_ratings[data["reviewerID"]].append(data["overall"])
        # for data in raw_val:
        #     user_ratings[data["reviewerID"]].append(data["overall"])
        # for data in raw_test:
        #     user_ratings[data["reviewerID"]].append(data["overall"])

        self.user_mean_rating = defaultdict(lambda: 2.5)
        self.user_mean_rating.update(
            {user_id: torch.tensor(ratings).mean() for user_id, ratings in user_ratings.items()}
        )

        self.train_dataset = [
            (
                int(self.user2id[data["reviewerID"]]) - 1,
                int(self.item2id[data["asin"]]) - 1,
                torch.tensor(data["overall"], dtype=torch.float32)
                - self.user_mean_rating[data["reviewerID"]],
            )
            for data in raw_train
        ]
        self.val_dataset = [
            (
                int(self.user2id[data["reviewerID"]]) - 1,
                int(self.item2id[data["asin"]]) - 1,
                torch.tensor(data["overall"], dtype=torch.float32),
            )
            for data in raw_val
        ]
        self.test_dataset = [
            (
                int(self.user2id[data["reviewerID"]]) - 1,
                int(self.item2id[data["asin"]]) - 1,
                torch.tensor(data["overall"], dtype=torch.float32),
            )
            for data in raw_test
        ]

        # Initialize model
        self.model = MF(self.num_users, self.num_items, self.num_factors)
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Initialize loss function
        self.loss_fn = nn.MSELoss()

    def train(self, use_best_model: bool = True):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        best_valid_loss = torch.inf
        best_valid_epoch = 0
        best_valid_model = None

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            dataloader_tqdm = tqdm(train_dataloader, desc=f"Train | Epoch {epoch}", ncols=80)
            for batch in dataloader_tqdm:
                user, item, rating = batch
                user = user.to(self.device)
                item = item.to(self.device)
                rating = rating.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                pred = self.model(user, item)
                loss = self.loss_fn(pred, rating)

                loss.backward()
                self.optimizer.step()

                dataloader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

            # Validate
            self.model.eval()
            dataloader_tqdm = tqdm(val_dataloader, desc=f"Validation | Epoch {epoch}", ncols=80)
            val_loss = 0.0

            with torch.no_grad():
                for batch in dataloader_tqdm:
                    user, item, rating = batch
                    user = user.to(self.device)
                    item = item.to(self.device)
                    rating = rating.to(self.device)

                    pred = self.model(user, item)
                    pred += torch.tensor(
                        [
                            self.user_mean_rating[self.id2user[str(user_id.item())]]
                            for user_id in user
                        ]
                    ).to(self.device)
                    loss = self.loss_fn(pred, rating)

                    val_loss += loss.item()

                    dataloader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

            val_loss /= len(val_dataloader)

            print(f"Validation loss: {val_loss:.4f}")

            if use_best_model and val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_valid_epoch = epoch

                self.model.to("cpu")
                best_valid_model = copy.deepcopy(self.model)
                self.model.to(self.device)

        if use_best_model:
            print(f"Best validation loss: {best_valid_loss:.4f} at epoch {best_valid_epoch}")
            self.model = best_valid_model

    def test(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        self.model.to(self.device)
        dataloader_tqdm = tqdm(test_dataloader, desc=f"Test", ncols=80)
        test_loss = 0.0

        with torch.no_grad():
            for batch in dataloader_tqdm:
                user, item, rating = batch
                user = user.to(self.device)
                item = item.to(self.device)
                rating = rating.to(self.device)

                pred = self.model(user, item)
                pred += torch.tensor(
                    [self.user_mean_rating[self.id2user[str(user_id.item())]] for user_id in user]
                ).to(self.device)
                loss = self.loss_fn(pred, rating)

                test_loss += loss.item()

                dataloader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

        test_loss /= len(test_dataloader)

        print(f"Test loss: {test_loss:.4f}")

    def save_embedding(self, save_path):
        embedding_dict: dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]] = dict()

        user_embedding = self.model.user_factors.weight.detach().clone().cpu()
        item_embedding = self.model.item_factors.weight.detach().clone().cpu()

        # Normalize embeddings to unit length
        user_embedding = normalize(user_embedding, dim=1)
        item_embedding = normalize(item_embedding, dim=1)

        # Save user and item embeddings
        embedding_dict["token_embedding"] = (
            torch.cat([user_embedding, item_embedding], dim=0)
        )

        # Save user and item embeddings separately
        embedding_dict["user_token_embedding"] = dict()
        for user_id in range(self.num_users):
            embedding_dict["user_token_embedding"]["user_" + str(user_id + 1)] = (
                user_embedding[user_id]
            )

        embedding_dict["item_token_embedding"] = dict()
        for item_id in range(self.num_items):
            embedding_dict["item_token_embedding"]["item_" + str(item_id + 1)] = (
                item_embedding[item_id]
            )

        pickle.dump(embedding_dict, open(save_path, "wb"))


def main(data_path: str):
    # Seed
    torch.manual_seed(306)
    torch.cuda.manual_seed(306)
    random.seed(306)
    np.random.seed(306)

    trainer = MFTrainer(
        device=torch.device("cuda"),
        num_factors=512,
        data_path=data_path,
        epochs=20,
        batch_size=512,
        lr=0.0001,
        weight_decay=0.01,
    )
    trainer.train()
    trainer.test()

    trainer.save_embedding(os.path.join("initialize", "beauty_mf.pkl"))


if __name__ == "__main__":
    main(data_path=os.path.join("data", "beauty"))
