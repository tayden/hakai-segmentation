import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from rasterio.windows import Window


# Implementation of paper:
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229839#pone.0229839.ref007


class Kernel(ABC):
    def __init__(self, size: int = 512):
        super().__init__()
        self.size = size
        self.wi = self._init_wi(size)
        self.wj = self.wi.copy()

    @staticmethod
    @abstractmethod
    def _init_wi(size: int) -> np.ndarray:
        raise NotImplementedError

    def get_kernel(
            self,
            top: bool = False,
            bottom: bool = False,
            left: bool = False,
            right: bool = False,
    ) -> "np.ndarray":
        wi, wj = self.wi.copy(), self.wj.copy()

        if top:
            wi[: self.size // 2] = 1
        if bottom:
            wi[self.size // 2:] = 1

        if left:
            wj[: self.size // 2] = 1
        if right:
            wj[self.size // 2:] = 1

        return np.expand_dims(wi, 1) @ np.expand_dims(wj, 0)

    def forward(
            self,
            x: "np.ndarray",
            top: bool = False,
            bottom: bool = False,
            left: bool = False,
            right: bool = False,
    ) -> np.ndarray:
        kernel = self.get_kernel(top=top, bottom=bottom, left=left, right=right)
        return np.multiply(x, kernel)


class HannKernel(Kernel):
    @staticmethod
    def _init_wi(size: int) -> np.ndarray:
        i = np.arange(0, size)
        return (1 - np.cos(((2 * np.pi * i) / (size - 1)))) / 2


class BartlettHannKernel(Kernel):
    @staticmethod
    def _init_wi(size: int) -> np.ndarray:
        # Follows original paper:
        # Ha YH, Pearce JA. A new window and comparison to standard windows.
        # IEEE Transactions on Acoustics, Speech, and Signal Processing.
        # 1989;37(2):298–301.
        i = np.arange(0, size)
        return (
                0.62
                - 0.48 * np.abs((i / size - 1 / 2))
                + 0.38 * np.cos((2 * np.pi * np.abs((i / size - 1 / 2))))
        )


class TriangularKernel(Kernel):
    @staticmethod
    def _init_wi(size: int) -> np.ndarray:
        i = np.arange(0, size)
        return 1 - np.abs((2 * i / size - 1))


class BlackmanKernel(Kernel):
    @staticmethod
    def _init_wi(size: int) -> np.ndarray:
        i = np.arange(0, size)
        return (
                0.42
                - 0.5 * np.cos((2 * np.pi * i / size))
                + 0.08 * np.cos((4 * np.pi * i / size))
        )


class NumpyMemoryRegister(object):
    def __init__(
            self,
            image_path: Union[str, Path],
            reg_depth: int,
            window_size: int,
    ):
        super().__init__()
        self.image_path = Path(image_path)
        self.n = reg_depth
        self.ws = window_size
        self.hws = window_size // 2

        # Copy metadata from img
        with rasterio.open(str(image_path), "r") as src:
            src_width = src.width

        self.height = self.ws
        self.width = (math.ceil(src_width / self.ws) * self.ws) + self.hws
        self.register = np.zeros((self.n, self.height, self.width))

    @property
    def _zero_chip(self):
        return np.zeros((self.n, self.hws, self.hws), dtype=float)

    def step(self, new_logits: "np.ndarray", img_window: Window):
        # 1. Read data from the registry to update with the new logits
        # |a|b| |
        # |c|d| |
        logits_abcd = self.register[:, :,
                      img_window.col_off: img_window.col_off + self.ws].copy()
        logits_abcd += new_logits

        # Update the registry and pop information-complete data
        # |c|b| | + pop a
        # |0|d| |
        logits_a = logits_abcd[:, : self.hws, : self.hws]
        logits_c = logits_abcd[:, self.hws:, : self.hws]
        logits_c0 = np.concatenate([logits_c, self._zero_chip], axis=1)
        logits_bd = logits_abcd[:, :, self.hws:]

        # write c0
        self.register[
        :, :, img_window.col_off: img_window.col_off + self.hws
        ] = logits_c0

        # write bd
        col_off_bd = img_window.col_off + self.hws
        self.register[:, :, col_off_bd: col_off_bd + self.hws] = logits_bd

        # Return the information-complete predictions
        preds_win = Window(
            col_off=img_window.col_off,
            row_off=img_window.row_off,
            height=min(self.hws, img_window.height),
            width=min(self.hws, img_window.width),
        )
        preds = logits_a[:, : img_window.height, : img_window.width]

        # Numpy softmax on axis 0
        preds = np.exp(preds) / np.sum(np.exp(preds), axis=0)

        return preds, preds_win
