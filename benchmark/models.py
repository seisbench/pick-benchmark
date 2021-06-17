import seisbench.models as sbm
import seisbench.generate as sbg

import pytorch_lightning as pl
import torch
import numpy as np
from abc import abstractmethod, ABC

# Allows to import this file in both jupyter notebook and code
try:
    from .augmentations import DuplicateEvent
except ImportError:
    from augmentations import DuplicateEvent


# Phase dict for labelling. We only study P and S phases without differentiating between them.
phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}


def vector_cross_entropy(y_pred, y_true, eps=1e-5):
    """
    Cross entropy loss

    :param y_true: True label probabilities
    :param y_pred: Predicted label probabilities
    :param eps: Epsilon to clip values for stability
    :return: Average loss across batch
    """
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h


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

    def get_train_augmentations(self):
        return self.get_augmentations()

    def get_val_augmentations(self):
        return self.get_augmentations()


class PhaseNetLit(SeisBenchModuleLit):
    """
    LightningModule for PhaseNet

    :param lr: Learning rate, defaults to 1e-2
    :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param kwargs: Kwargs are passed to the SeisBench.models.PhaseNet constructor.
    """

    def __init__(self, lr=1e-2, sigma=10, sample_boundaries=(None, None), **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.sigma = sigma
        self.sample_boundaries = sample_boundaries
        self.loss = vector_cross_entropy
        self.model = sbm.PhaseNet(**kwargs)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x = batch["X"]
        y_true = batch["y"]
        y_pred = self.model(x)
        return self.loss(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_augmentations(self):
        return [
            # In 80 % of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 20 % of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=3000,
                        windowlen=6000,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[4, 1],
            ),
            sbg.RandomWindow(
                low=self.sample_boundaries[0],
                high=self.sample_boundaries[1],
                windowlen=3001,
                strategy="pad",
            ),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            sbg.ProbabilisticLabeller(
                label_columns=phase_dict, sigma=self.sigma, dim=0
            ),
        ]


class GPDLit(SeisBenchModuleLit):
    """
    LightningModule for PhaseNet

    :param lr: Learning rate, defaults to 1e-2
    :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param kwargs: Kwargs are passed to the SeisBench.models.PhaseNet constructor.
    """

    def __init__(self, lr=1e-3, highpass=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = sbm.GPD(**kwargs)
        self.loss = torch.nn.NLLLoss()
        self.highpass = highpass

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x = batch["X"]
        y_true = batch["y"].squeeze()
        y_pred = torch.log(self.model(x))
        return self.loss(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_augmentations(self):
        filter = []
        if self.highpass is not None:
            filter = [sbg.Filter(1, self.highpass, "highpass")]

        return (
            [
                # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
                # Uses strategy variable, as padding will be handled by the random window.
                # In 1/3 of the cases, just returns the original trace, to keep diversity high.
                sbg.OneOf(
                    [
                        sbg.WindowAroundSample(
                            list(phase_dict.keys()),
                            samples_before=400,
                            windowlen=800,
                            selection="random",
                            strategy="variable",
                        ),
                        sbg.NullAugmentation(),
                    ],
                    probabilities=[2, 1],
                ),
                sbg.RandomWindow(
                    windowlen=400,
                    strategy="pad",
                ),
                sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
                sbg.StandardLabeller(
                    label_columns=phase_dict,
                    on_overlap="fixed-relevance",
                    low=100,
                    high=-100,
                ),
            ]
            + filter
            + [sbg.ChangeDtype(np.float32)]
        )


class EQTransformerLit(SeisBenchModuleLit):
    """
    LightningModule for EQTransformer

    :param lr: Learning rate, defaults to 1e-2
    :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param kwargs: Kwargs are passed to the SeisBench.models.EQTransformer constructor.
    """

    def __init__(
        self,
        lr=1e-2,
        sigma=10,
        sample_boundaries=(None, None),
        loss_weights=(0.05, 0.40, 0.55),
        rotate_array=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.sigma = sigma
        self.sample_boundaries = sample_boundaries
        self.loss = torch.nn.BCELoss()
        self.loss_weights = loss_weights
        self.rotate_array = rotate_array
        self.model = sbm.EQTransformer(**kwargs)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x = batch["X"]
        p_true = batch["y"][:, 0]
        s_true = batch["y"][:, 1]
        det_true = batch["detections"][:, 0]
        det_pred, p_pred, s_pred = self.model(x)

        return (
            self.loss_weights[0] * self.loss(det_pred, det_true)
            + self.loss_weights[1] * self.loss(p_pred, p_true)
            + self.loss_weights[2] * self.loss(s_pred, s_true)
        )

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_joint_augmentations(self):
        p_phases = [key for key, val in phase_dict.items() if val == "P"]
        s_phases = [key for key, val in phase_dict.items() if val == "S"]

        block1 = [
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=6000,
                        windowlen=12000,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                low=self.sample_boundaries[0],
                high=self.sample_boundaries[1],
                windowlen=6000,
                strategy="pad",
            ),
            sbg.ProbabilisticLabeller(
                label_columns=phase_dict, sigma=self.sigma, dim=0
            ),
            sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections")),
            # Normalize to ensure correct augmentation behavior
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

        block2 = [
            sbg.ChangeDtype(np.float32, "X"),
            sbg.ChangeDtype(np.float32, "y"),
            sbg.ChangeDtype(np.float32, "detections"),
        ]

        return block1, block2

    def get_train_augmentations(self):
        if self.rotate_array:
            rotation_block = [
                sbg.OneOf(
                    [
                        sbg.RandomArrayRotation(["X", "y", "detections"]),
                        sbg.NullAugmentation(),
                    ],
                    [0.99, 0.01],
                )
            ]
        else:
            rotation_block = []

        augmentation_block = [
            # Add secondary event
            sbg.OneOf(
                [DuplicateEvent(label_keys="y"), sbg.NullAugmentation()],
                probabilities=[0.3, 0.7],
            ),
            # Gaussian noise
            sbg.OneOf([sbg.GaussianNoise(), sbg.NullAugmentation()], [0.5, 0.5]),
            # Array rotation
            *rotation_block,
            # Gaps
            sbg.OneOf([sbg.AddGap(), sbg.NullAugmentation()], [0.2, 0.8]),
            # Channel dropout
            sbg.OneOf([sbg.ChannelDropout(), sbg.NullAugmentation()], [0.3, 0.7]),
            # Augmentations make second normalize necessary
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

        block1, block2 = self.get_joint_augmentations()

        return block1 + augmentation_block + block2

    def get_val_augmentations(self):
        block1, block2 = self.get_joint_augmentations()

        return block1 + block2

    def get_augmentations(self):
        raise NotImplementedError("Use get_train/val_augmentations instead.")


class CREDLit(SeisBenchModuleLit):
    """
    LightningModule for CRED

    :param lr: Learning rate, defaults to 1e-2
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param kwargs: Kwargs are passed to the SeisBench.models.CRED constructor.
    """

    def __init__(self, lr=1e-2, sample_boundaries=(None, None), **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.sample_boundaries = sample_boundaries
        self.loss = torch.nn.BCELoss()
        self.model = sbm.CRED(**kwargs)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x = batch["spec"]
        y_true = batch["y"][:, 0]
        y_pred = self.model(x)[:, :, 0]

        return self.loss(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_augmentations(self):
        p_phases = [key for key, val in phase_dict.items() if val == "P"]
        s_phases = [key for key, val in phase_dict.items() if val == "S"]

        def spectrogram(state_dict):
            x, metadata = state_dict["X"]
            spec = self.model.waveforms_to_spectrogram(x)
            state_dict["spec"] = (spec, metadata)

        def resample_detections(state_dict):
            # Resample detections to 19 samples as in the output of CRED
            y, metadata = state_dict["y"]
            state_dict["y"] = (y[:, ::158], metadata)

        augmentations = [
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=3000,
                        windowlen=6000,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                low=self.sample_boundaries[0],
                high=self.sample_boundaries[1],
                windowlen=3000,
                strategy="pad",
            ),
            sbg.DetectionLabeller(p_phases, s_phases),
            # Normalize to ensure correct augmentation behavior
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            spectrogram,
            resample_detections,
            sbg.ChangeDtype(np.float32, "y"),
            sbg.ChangeDtype(np.float32, "spec"),
        ]

        return augmentations
