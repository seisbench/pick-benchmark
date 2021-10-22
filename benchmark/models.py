"""
This file contains the model specifications.
"""

import seisbench.models as sbm
import seisbench.generate as sbg

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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
    if y_pred.ndim == 3:
        h = h.mean(-1).sum(
            -1
        )  # Mean along sample dimension and sum along pick dimension
    else:
        h = h.sum(-1)  # Sum along pick dimension
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
        """
        Returns the set of training augmentations.
        """
        return self.get_augmentations()

    def get_val_augmentations(self):
        """
        Returns the set of validation augmentations for validations during training.
        """
        return self.get_augmentations()

    @abstractmethod
    def get_eval_augmentations(self):
        """
        Returns the set of evaluation augmentations for evaluation after training.
        These augmentations will be passed to a SteeredGenerator and should usually contain a steered window.
        """
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        """
        Predict step for the lightning module. Returns results for three tasks:

        - earthquake detection (score, higher means more likely detection)
        - P to S phase discrimination (score, high means P, low means S)
        - phase location in samples (two integers, first for P, second for S wave)

        All predictions should only take the window defined by batch["window_borders"] into account.

        :param batch:
        :return:
        """
        score_detection = None
        score_p_or_s = None
        p_sample = None
        s_sample = None
        return score_detection, score_p_or_s, p_sample, s_sample


class PhaseNetLit(SeisBenchModuleLit):
    """
    LightningModule for PhaseNet

    :param lr: Learning rate, defaults to 1e-2
    :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param kwargs: Kwargs are passed to the SeisBench.models.PhaseNet constructor.
    """

    def __init__(self, lr=1e-2, sigma=20, sample_boundaries=(None, None), **kwargs):
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
                windowlen=3001,
                strategy="pad",
            ),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            sbg.ProbabilisticLabeller(
                label_columns=phase_dict, sigma=self.sigma, dim=0
            ),
        ]

    def get_eval_augmentations(self):
        return [
            sbg.SteeredWindow(windowlen=3001, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        window_borders = batch["window_borders"]

        pred = self.model(x)

        score_detection = torch.zeros(pred.shape[0])
        score_p_or_s = torch.zeros(pred.shape[0])
        p_sample = torch.zeros(pred.shape[0], dtype=int)
        s_sample = torch.zeros(pred.shape[0], dtype=int)

        for i in range(pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            local_pred = pred[i, :, start_sample:end_sample]

            score_detection[i] = torch.max(1 - local_pred[-1])  # 1 - noise
            score_p_or_s[i] = torch.max(local_pred[0]) / torch.max(
                local_pred[1]
            )  # most likely P by most likely S

            p_sample[i] = torch.argmax(local_pred[0])
            s_sample[i] = torch.argmax(local_pred[1])

        return score_detection, score_p_or_s, p_sample, s_sample


class GPDLit(SeisBenchModuleLit):
    """
    LightningModule for GPD

    :param lr: Learning rate, defaults to 1e-3
    :param sigma: Standard deviation passed to the ProbabilisticPickLabeller. If not, uses determinisic labels,
                  i.e., whether a pick is contained.
    :param highpass: If not None, cutoff frequency for highpass filter in Hz.
    :param kwargs: Kwargs are passed to the SeisBench.models.GPD constructor.
    """

    def __init__(self, lr=1e-3, highpass=None, sigma=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.sigma = sigma
        self.model = sbm.GPD(**kwargs)
        if sigma is None:
            self.nllloss = torch.nn.NLLLoss()
            self.loss = self.nll_with_probabilities
        else:
            self.loss = vector_cross_entropy
        self.highpass = highpass
        self.predict_stride = 5

    def nll_with_probabilities(self, y_pred, y_true):
        y_pred = torch.log(y_pred)
        return self.nllloss(y_pred, y_true)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x = batch["X"]
        y_true = batch["y"].squeeze()
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
        filter = []
        if self.highpass is not None:
            filter = [sbg.Filter(1, self.highpass, "highpass")]

        if self.sigma is None:
            labeller = sbg.StandardLabeller(
                label_columns=phase_dict,
                on_overlap="fixed-relevance",
                low=100,
                high=-100,
            )
        else:
            labeller = sbg.ProbabilisticPointLabeller(
                label_columns=phase_dict, position=0.5, sigma=self.sigma
            )

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
                labeller,
            ]
            + filter
            + [sbg.ChangeDtype(np.float32)]
        )

    def get_eval_augmentations(self):
        filter = []
        if self.highpass is not None:
            filter = [sbg.Filter(1, self.highpass, "highpass")]

        return [
            # Larger window length ensures a sliding window covering full trace can be applied
            sbg.SteeredWindow(windowlen=3400, strategy="pad"),
            sbg.SlidingWindow(timestep=self.predict_stride, windowlen=400),
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            *filter,
            sbg.ChangeDtype(np.float32),
        ]

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        window_borders = batch["window_borders"]

        shape_save = x.shape
        x = x.reshape(
            (-1,) + shape_save[2:]
        )  # Merge batch and sliding window dimensions
        pred = self.model(x)
        pred = pred.reshape(shape_save[:2] + (-1,))
        pred = torch.repeat_interleave(
            pred, self.predict_stride, dim=1
        )  # Counteract stride
        pred = F.pad(pred, (0, 0, 200, 200))
        pred = pred.permute(0, 2, 1)

        # Otherwise windows shorter 30 s will automatically produce detections
        pred[:, 2, :200] = 1
        pred[:, 2, -200:] = 1

        score_detection = torch.zeros(pred.shape[0])
        score_p_or_s = torch.zeros(pred.shape[0])
        p_sample = torch.zeros(pred.shape[0], dtype=int)
        s_sample = torch.zeros(pred.shape[0], dtype=int)

        for i in range(pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            local_pred = pred[i, :, start_sample:end_sample]

            score_detection[i] = torch.max(1 - local_pred[-1])  # 1 - noise
            score_p_or_s[i] = torch.max(local_pred[0]) / torch.max(
                local_pred[1]
            )  # most likely P by most likely S

            # Adjust for prediction stride by choosing the sample in the middle of each block
            p_sample[i] = torch.argmax(local_pred[0]) + self.predict_stride // 2
            s_sample[i] = torch.argmax(local_pred[1]) + self.predict_stride // 2

        return score_detection, score_p_or_s, p_sample, s_sample


class EQTransformerLit(SeisBenchModuleLit):
    """
    LightningModule for EQTransformer

    :param lr: Learning rate, defaults to 1e-2
    :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param loss_weights: Loss weights for detection, P and S phase.
    :param rotate_array: If true, rotate array along sample axis.
    :param detection_fixed_window: Passed as parameter fixed_window to detection
    :param kwargs: Kwargs are passed to the SeisBench.models.EQTransformer constructor.
    """

    def __init__(
        self,
        lr=1e-2,
        sigma=20,
        sample_boundaries=(None, None),
        loss_weights=(0.05, 0.40, 0.55),
        rotate_array=False,
        detection_fixed_window=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.sigma = sigma
        self.sample_boundaries = sample_boundaries
        self.loss = torch.nn.BCELoss()
        self.loss_weights = loss_weights
        self.rotate_array = rotate_array
        self.detection_fixed_window = detection_fixed_window
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

        if self.detection_fixed_window is not None:
            detection_labeller = sbg.DetectionLabeller(
                p_phases,
                fixed_window=self.detection_fixed_window,
                key=("X", "detections"),
            )
        else:
            detection_labeller = sbg.DetectionLabeller(
                p_phases, s_phases=s_phases, key=("X", "detections")
            )

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
            detection_labeller,
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

    def get_eval_augmentations(self):
        return [
            sbg.SteeredWindow(windowlen=6000, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        window_borders = batch["window_borders"]

        det_pred, p_pred, s_pred = self.model(x)

        score_detection = torch.zeros(det_pred.shape[0])
        score_p_or_s = torch.zeros(det_pred.shape[0])
        p_sample = torch.zeros(det_pred.shape[0], dtype=int)
        s_sample = torch.zeros(det_pred.shape[0], dtype=int)

        for i in range(det_pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            local_det_pred = det_pred[i, start_sample:end_sample]
            local_p_pred = p_pred[i, start_sample:end_sample]
            local_s_pred = s_pred[i, start_sample:end_sample]

            score_detection[i] = torch.max(local_det_pred)
            score_p_or_s[i] = torch.max(local_p_pred) / torch.max(
                local_s_pred
            )  # most likely P by most likely S

            p_sample[i] = torch.argmax(local_p_pred)
            s_sample[i] = torch.argmax(local_s_pred)

        return score_detection, score_p_or_s, p_sample, s_sample


class CREDLit(SeisBenchModuleLit):
    """
    LightningModule for CRED

    :param lr: Learning rate, defaults to 1e-2
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param kwargs: Kwargs are passed to the SeisBench.models.CRED constructor.
    """

    def __init__(
        self,
        lr=1e-2,
        sample_boundaries=(None, None),
        detection_fixed_window=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.sample_boundaries = sample_boundaries
        self.detection_fixed_window = detection_fixed_window
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
            # Each sample represents the average over 158 original samples
            y, metadata = state_dict["y"]
            y = np.pad(y, [(0, 0), (0, 2)], mode="constant", constant_values=0)
            y = np.reshape(y, (1, 19, 158))
            y = np.mean(y, axis=-1)
            state_dict["y"] = (y, metadata)

        if self.detection_fixed_window is not None:
            detection_labeller = sbg.DetectionLabeller(
                p_phases, fixed_window=self.detection_fixed_window
            )
        else:
            detection_labeller = sbg.DetectionLabeller(p_phases, s_phases=s_phases)

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
            detection_labeller,
            # Normalize to ensure correct augmentation behavior
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            spectrogram,
            resample_detections,
            sbg.ChangeDtype(np.float32, "y"),
            sbg.ChangeDtype(np.float32, "spec"),
        ]

        return augmentations

    def get_eval_augmentations(self):
        def spectrogram(state_dict):
            x, metadata = state_dict["X"]
            spec = self.model.waveforms_to_spectrogram(x)
            state_dict["spec"] = (spec, metadata)

        return [
            sbg.SteeredWindow(windowlen=3000, strategy="pad"),
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            spectrogram,
            sbg.ChangeDtype(np.float32, "spec"),
        ]

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["spec"]
        window_borders = batch["window_borders"]

        pred = self.model(x)

        score_detection = torch.zeros(pred.shape[0])
        score_p_or_s = torch.zeros(pred.shape[0]) * np.nan
        p_sample = torch.zeros(pred.shape[0], dtype=int) * np.nan
        s_sample = torch.zeros(pred.shape[0], dtype=int) * np.nan

        for i in range(pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            # We go for a slightly broader window, i.e., all output prediction points encompassing the target window
            start_resampled = start_sample // 158
            end_resampled = end_sample // 158 + 1
            local_pred = pred[i, start_resampled:end_resampled, 0]

            score_detection[i] = torch.max(local_pred)  # 1 - noise

        return score_detection, score_p_or_s, p_sample, s_sample


class BasicPhaseAELit(SeisBenchModuleLit):
    """
    LightningModule for BasicPhaseAE

    :param lr: Learning rate, defaults to 1e-2
    :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param kwargs: Kwargs are passed to the SeisBench.models.BasicPhaseAE constructor.
    """

    def __init__(self, lr=1e-2, sigma=20, sample_boundaries=(None, None), **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.sigma = sigma
        self.sample_boundaries = sample_boundaries
        self.loss = vector_cross_entropy
        self.model = sbm.BasicPhaseAE(**kwargs)

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
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=700,
                        windowlen=700,
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
                windowlen=600,
                strategy="pad",
            ),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            sbg.ProbabilisticLabeller(
                label_columns=phase_dict, sigma=self.sigma, dim=0
            ),
        ]

    def get_eval_augmentations(self):
        return [
            sbg.SteeredWindow(windowlen=3000, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        window_borders = batch["window_borders"]

        # Create overlapping windows
        re = torch.zeros(x.shape[:2] + (7, 600), dtype=x.dtype, device=x.device)
        for i, start in enumerate(range(0, 2401, 400)):
            re[:, :, i] = x[:, :, start : start + 600]
        x = re

        x = x.permute(0, 2, 1, 3)  # --> (batch, windows, channels, samples)
        shape_save = x.shape
        x = x.reshape(-1, 3, 600)  # --> (batch * windows, channels, samples)
        window_pred = self.model(x)
        window_pred = window_pred.reshape(
            shape_save[:2] + (3, 600)
        )  # --> (batch, window, channels, samples)

        # Connect predictions again, ignoring first and last second of each prediction
        pred = torch.zeros((window_pred.shape[0], window_pred.shape[2], 3000))
        for i, start in enumerate(range(0, 2401, 400)):
            if start == 0:
                # Use full window (for start==0, the end will be overwritten)
                pred[:, :, start : start + 600] = window_pred[:, i]
            else:
                pred[:, :, start + 100 : start + 600] = window_pred[:, i, :, 100:]

        score_detection = torch.zeros(pred.shape[0])
        score_p_or_s = torch.zeros(pred.shape[0])
        p_sample = torch.zeros(pred.shape[0], dtype=int)
        s_sample = torch.zeros(pred.shape[0], dtype=int)

        for i in range(pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            local_pred = pred[i, :, start_sample:end_sample]

            score_detection[i] = torch.max(1 - local_pred[-1])  # 1 - noise
            score_p_or_s[i] = torch.max(local_pred[0]) / torch.max(
                local_pred[1]
            )  # most likely P by most likely S

            p_sample[i] = torch.argmax(local_pred[0])
            s_sample[i] = torch.argmax(local_pred[1])

        return score_detection, score_p_or_s, p_sample, s_sample


class DPPDetectorLit(SeisBenchModuleLit):
    """
    LightningModule for DPPDetector

    :param lr: Learning rate, defaults to 1e-2
    :param kwargs: Kwargs are passed to the SeisBench.models.PhaseNet constructor.
    """

    def __init__(self, lr=1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.nllloss = torch.nn.NLLLoss()
        self.loss = self.nll_with_probabilities
        self.model = sbm.DPPDetector(**kwargs)

    def forward(self, x):
        return self.model(x)

    def nll_with_probabilities(self, y_pred, y_true):
        y_pred = torch.log(y_pred + 1e-5)
        return self.nllloss(y_pred, y_true)

    def shared_step(self, batch):
        x = batch["X"]
        y_true = batch["y"].squeeze()
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
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=650,
                        windowlen=1300,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                windowlen=500,
                strategy="pad",
            ),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            sbg.StandardLabeller(
                label_columns=phase_dict, on_overlap="fixed-relevance"
            ),
        ]

    def get_eval_augmentations(self):
        return [
            sbg.SteeredWindow(windowlen=3000, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        window_borders = batch["window_borders"]

        # Create windows
        x = x.reshape(x.shape[:-1] + (6, 500))  # Split into 6 windows of length 500
        x = x.permute(0, 2, 1, 3)  # --> (batch, windows, channels, samples)
        shape_save = x.shape
        x = x.reshape(-1, 3, 500)  # --> (batch * windows, channels, samples)
        pred = self.model(x)
        pred = pred.reshape(shape_save[:2] + (-1,))  # --> (batch, windows, label)

        score_detection = torch.zeros(pred.shape[0])
        score_p_or_s = torch.zeros(pred.shape[0])
        p_sample = torch.zeros(pred.shape[0], dtype=int) * np.nan
        s_sample = torch.zeros(pred.shape[0], dtype=int) * np.nan

        for i in range(pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            start_resampled = start_sample.cpu() // 500
            end_resampled = int(np.ceil(end_sample.cpu() / 500))
            local_pred = pred[i, start_resampled:end_resampled, :]

            score_p_or_s[i] = torch.max(local_pred[:, 0]) / torch.max(local_pred[:, 1])
            score_detection[i] = torch.max(1 - local_pred[:, -1])

        return score_detection, score_p_or_s, p_sample, s_sample


class DPPPickerLit(SeisBenchModuleLit):
    """
    LightningModule for DPPPicker

    :param lr: Learning rate, defaults to 1e-2
    :param kwargs: Kwargs are passed to the SeisBench.models.PhaseNet constructor.
    """

    def __init__(self, mode, lr=1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.lr = lr
        self.loss = torch.nn.BCELoss()
        self.model = sbm.DPPPicker(mode=mode, **kwargs)

    def forward(self, x):
        if self.mode == "P":
            x = x[:, 0:1]  # Select vertical component
        elif self.mode == "S":
            x = x[:, 1:3]  # Select horizontal components
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return self.model(x)

    def shared_step(self, batch):
        x = batch["X"]
        y_true = batch["y"]

        if self.mode == "P":
            y_true = y_true[:, 0]  # P wave
            x = x[:, 0:1]  # Select vertical component
        elif self.mode == "S":
            y_true = y_true[:, 1]  # S wave
            x = x[:, 1:3]  # Select horizontal components
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        y_pred = self.model(x)

        loss = self.loss(y_pred, y_true)
        return loss

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
            sbg.WindowAroundSample(
                [key for key, val in phase_dict.items() if val == self.mode],
                samples_before=1000,
                windowlen=2000,
                selection="random",
                strategy="variable",
            ),
            sbg.RandomWindow(
                windowlen=1000,
                strategy="pad",
            ),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            sbg.StepLabeller(label_columns=phase_dict),
            sbg.ChangeDtype(np.float32),
            sbg.ChangeDtype(np.float32, "y"),
        ]

    def get_eval_augmentations(self):
        return [
            sbg.SteeredWindow(windowlen=1000, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        window_borders = batch["window_borders"]

        pred = self(x)

        score_detection = torch.zeros(pred.shape[0]) * np.nan
        score_p_or_s = torch.zeros(pred.shape[0]) * np.nan
        p_sample = torch.zeros(pred.shape[0], dtype=int) * np.nan
        s_sample = torch.zeros(pred.shape[0], dtype=int) * np.nan

        for i in range(pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            local_pred = pred[i, start_sample:end_sample]

            if (local_pred > 0.5).any():
                sample = torch.argmax(
                    (local_pred > 0.5).float()
                )  # First sample exceeding 0.5
            else:
                sample = 500  # Simply guess the middle

            if self.mode == "P":
                p_sample[i] = sample
            elif self.mode == "S":
                s_sample[i] = sample
            else:
                raise ValueError(f"Unknown mode {self.mode}")

        return score_detection, score_p_or_s, p_sample, s_sample
