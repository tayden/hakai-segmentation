import gc
from pathlib import Path
from typing import Optional, Union

import torch

from hakai_segmentation.types import ModelTypeT
from hakai_segmentation.weights import get_model_weights


class SegmentationModel(object):
    def __init__(
        self,
        model_type: ModelTypeT,
        weights_version: Optional[str] = None,
        use_gpu: bool = True,
    ):
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and use_gpu
            else torch.device("cpu")
        )
        self.model_type = model_type

        self.weights = get_model_weights(model_type, weights_version)
        self.weights.download_if_missing()
        self.model = self.load_model(self.weights.path)

    def load_model(self, model_weights: Union[Path, str]) -> "torch.nn.Module":
        model = torch.jit.load(model_weights, map_location=self.device)
        model.eval()
        return model

    def reload(self):
        del self.model
        gc.collect()
        self.model = self.load_model()

    def __call__(self, batch: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            return self.model.forward(batch.to(self.device))
