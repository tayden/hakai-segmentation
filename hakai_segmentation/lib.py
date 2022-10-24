from typing import Optional

from hakai_segmentation.models import SegmentationModel
from hakai_segmentation.process import GeotiffSegmentation
from hakai_segmentation.types import ModelTypeT
from hakai_segmentation.weights import get_available_model_weights


def list_versions(model_type: Optional[ModelTypeT] = None):
    """Get the available model versions with some metadata."""
    return get_available_model_weights(model_type)


def find_kelp(
    source: str,
    dest: str,
    species: bool = False,
    crop_size: int = 256,
    padding: int = 128,
    batch_size: int = 2,
    use_gpu: bool = True,
    weights_version: Optional[str] = None,
):
    """Detect kelp in image at path `source` and output the resulting classification raster to file at path `dest`.

    :param source: Input image with Byte data type.
    :param dest: File path location to save output to.
    :param species: Do species classification instead of presence/absence.
    :param crop_size: The size of cropped image square run through the segmentation model.
    :param padding: The number of context pixels added to each side of the cropped image squares.
    :param batch_size: The batch size of cropped image sections to process together.
    :param use_gpu: Disable Cuda GPU usage and run on CPU only.
    :param weights_version: Optionally specify weights version str. See `list_versions()` for options.
    """
    if species:
        model = SegmentationModel(
            model_type="kelp_species", weights_version=weights_version, use_gpu=use_gpu
        )
    else:
        model = SegmentationModel(
            model_type="kelp_presence", weights_version=weights_version, use_gpu=use_gpu
        )
    GeotiffSegmentation(
        model, source, dest, crop_size=crop_size, padding=padding, batch_size=batch_size
    )()


def find_mussels(
    source: str,
    dest: str,
    crop_size: int = 256,
    padding: int = 128,
    batch_size: int = 2,
    use_gpu: bool = True,
    weights_version: Optional[str] = None,
):
    """Detect mussels in image at path `source` and output the resulting classification raster to file at path `dest`.

    :param source: Input image with Byte data type.
    :param dest: File path location to save output to.
    :param crop_size: The size of cropped image square run through the segmentation model.
    :param padding: The number of context pixels added to each side of the cropped image squares.
    :param batch_size: The batch size of cropped image sections to process together.
    :param use_gpu: Disable Cuda GPU usage and run on CPU only.
    :param weights_version: Optionally specify weights version str. See `kom versions` for options.
    """
    model = SegmentationModel(
        model_type="mussel_presence", weights_version=weights_version, use_gpu=use_gpu
    )
    GeotiffSegmentation(
        model, source, dest, crop_size=crop_size, padding=padding, batch_size=batch_size
    )()
