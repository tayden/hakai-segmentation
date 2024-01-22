from abc import ABC, abstractmethod, ABCMeta
from typing import Union, Type

import numpy as np
import onnxruntime as ort
from PIL.Image import Image

from kelp_o_matic.utils import lazy_load_params


class _Model(ABC):
    register_depth = 2

    @staticmethod
    def transform(x: Union[np.ndarray, Image]) -> "np.ndarray":
        x = np.asarray(x).astype(np.float32)
        x = np.moveaxis(x, 0, -1)

        # Mean/Std scaling
        x -= np.array([0.485, 0.456, 0.406])
        x /= np.array([0.229, 0.224, 0.225])
        return x

    def __init__(self, *args, **kwargs):
        self.model = self.load_model()

    @property
    @abstractmethod
    def onnx_path(self) -> str:
        raise NotImplementedError

    def load_model(self) -> "ort.InferenceSession":
        params_file = lazy_load_params(self.onnx_path)
        ort_session = ort.InferenceSession(
            params_file,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        )
        return ort_session

    def __call__(self, x: "np.ndarray") -> "np.ndarray":
        output_name = self.model.get_outputs()[0].name
        input_name = self.model.get_inputs()[0].name
        onnx_pred = self.model.run([output_name], {input_name: x})
        return onnx_pred[0]

    def post_process(self, x: "np.ndarray") -> "np.ndarray":
        return x.argmax(axis=0)

    def shortcut(self, crop_size: int):
        """Shortcut prediction for when we know a cropped section is background.
        Prevent unnecessary forward passes through model."""
        logits = np.zeros((self.register_depth, crop_size, crop_size))
        logits[0] = 1
        return logits


class _SpeciesSegmentationModel(_Model, metaclass=ABCMeta):
    register_depth = 4

    @property
    @abstractmethod
    def presence_model_class(self) -> Type["_Model"]:
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.presence_model = self.presence_model_class(*args, **kwargs)

    def __call__(self, x: "np.ndarray") -> "np.ndarray":
        presence_logits = self.presence_model(x)  # 0: bg, 1: kelp

        output_name = self.model.get_outputs()[0].name
        input_name = self.model.get_inputs()[0].name
        onnx_pred = self.model.run([output_name], {input_name: x})
        species_logits = onnx_pred[0]  # 0: macro, 1: nerea
        logits = np.concatenate((presence_logits, species_logits), axis=1)

        return logits  # [[0: bg, 1: kelp], [0: macro, 1: nereo]]

    def post_process(self, x: "np.ndarray") -> "np.ndarray":
        presence = np.argmax(x[:2], axis=0)  # 0: bg, 1: kelp
        species = np.argmax(x[2:], axis=0) + 2  # 2: macro, 3: nereo
        label = np.multiply(presence, species)  # 0: bg, 2: macro, 3: nereo
        return label


class KelpRGBPresenceSegmentationModel(_Model):
    onnx_path = "LRASPP_MobileNetV3_kelp_presence_rgb_miou=0.8023.onnx"


class KelpRGBSpeciesSegmentationModel(_SpeciesSegmentationModel):
    onnx_path = "LRASPP_MobileNetV3_kelp_species_rgb_miou=0.9634.onnx"
    presence_model_class = KelpRGBPresenceSegmentationModel


class MusselRGBPresenceSegmentationModel(_Model):
    onnx_path = "LRASPP_MobileNetV3_mussel_presence_rgb_miou=0.8745.onnx"


def _unet_efficientnet_b4_transform(x: Union[np.ndarray, Image]) -> "np.ndarray":
    x = np.asarray(x[:, :, :4]).astype(np.float32)
    x = np.moveaxis(x, -1,0)

    # min-max scale
    # get unique values
    x_unique = np.unique(x.flatten())
    x_unique.sort()
    min_ = x_unique[0]
    if len(x_unique) > 1:
        min_ = x_unique[1]
    max_ = x_unique[-1]
    return np.clip((x - min_) / (max_ - min_ + 1e-8), 0, 1)


class KelpRGBIPresenceSegmentationModel(_Model):
    onnx_path = (
        "UNetPlusPlus_EfficientNetB4_kelp_presence_aco_rgbi_miou=0.8785.onnx"
    )

    @staticmethod
    def transform(x: Union[np.ndarray, Image]) -> np.ndarray:
        return _unet_efficientnet_b4_transform(x)


class KelpRGBISpeciesSegmentationModel(_SpeciesSegmentationModel):
    onnx_path = (
        "UNetPlusPlus_EfficientNetB4_kelp_species_rgbi_miou=0.8432.onnx"
    )
    presence_model_class = KelpRGBIPresenceSegmentationModel

    @staticmethod
    def transform(x: Union[np.ndarray, Image]) -> np.ndarray:
        return _unet_efficientnet_b4_transform(x)
