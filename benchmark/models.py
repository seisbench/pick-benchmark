import seisbench.models as sbm
import seisbench.generate as sbg

import pytorch_lightning as pl
import torch
from abc import abstractmethod, ABC


class SeisBenchModuleLit(pl.LightningModule, ABC):
    """
    Abstract interface for SeisBench lightning modules.
    Adds generic function, e.g., get_augmentations
    """

    @abstractmethod
    def get_augmentations(self):
        """
        Returns a list of augmentations that can be passed to the seisbench.generate.GenericGenerator

        :return: List of augmentations
        """
        pass


class PhaseNetLit(pl.LightningModule):
    def __init__(self, lr=1e-2, **kwargs):
        super().__init__()
        self.lr = lr
        self.model = sbm.PhaseNet(**kwargs)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x = batch["X"]
        y_true = batch["y"]
        y_pred = self.model(x)
        return self.loss(y_true, y_pred)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)
        return loss

    @staticmethod
    def loss(y_true, y_pred, eps=1e-5):
        """
        Cross entropy loss

        :param y_true: True label probabilities
        :param y_pred: Predicted label probabilities
        :param eps: Epsilon to clip values for stability
        :return: Average loss across batch
        """
        h = y_true * torch.log(y_pred + eps)
        h = h.mean(-1).sum(
            -1
        )  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis
        return -h

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_augmentations(self):
        pick_labels = ["trace_p_arrival_sample", "trace_s_arrival_sample"]
        return [
            sbg.RandomWindow(windowlen=3001, strategy="pad"),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            sbg.ProbabilisticLabeller(label_columns=pick_labels, sigma=10, dim=0),
        ]
