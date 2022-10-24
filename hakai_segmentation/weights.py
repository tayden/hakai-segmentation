from pathlib import Path
from typing import Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from rich import print

from hakai_segmentation.types import ModelTypeT

_BUCKET_NAME = "kelp-o-matic"
_WEIGHTS_DIR = "~/.kom"


class ModelWeights:
    """Represents a single set of versioned model parameters for a segmentation model."""

    def __init__(self, object_version: "boto3.resources.factory.s3.ObjectVersion"):
        self.obj = object_version
        self.weights_dir = Path(_WEIGHTS_DIR).expanduser()

    @property
    def path(self) -> Path:
        return self.weights_dir.joinpath(f"{self.model_type}_{self.obj.version_id}")

    def download_if_missing(self):
        self.weights_dir.mkdir(exist_ok=True)

        # TODO: Compare checksums to ensure integrity
        if not self.path.is_file():
            print(f"Downloading model {self.path}")

            client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            client.download_file(
                _BUCKET_NAME,
                self.key,
                str(self.path),
                ExtraArgs={"VersionId": self.version_id},
            )

    @property
    def key(self):
        return self.obj.key

    @property
    def model_type(self):
        return self.obj.key[: -len("_jit.pt")]

    @property
    def version_id(self):
        return self.obj.version_id

    @property
    def is_latest(self):
        return self.obj.is_latest

    @property
    def last_modified(self):
        return self.obj.last_modified

    @property
    def size(self):
        return self.obj.size

    def __repr__(self):
        return str(
            dict(
                key=self.obj.key,
                model_type=self.model_type,
                version_id=self.obj.version_id,
                is_latest=self.obj.is_latest,
                last_modified=self.obj.last_modified,
                size=self.obj.size,
            )
        )


def get_available_model_weights(model_type: Optional[ModelTypeT] = None):
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(_BUCKET_NAME)

    if model_type:
        return [
            ModelWeights(a) for a in bucket.object_versions.filter(Prefix=model_type)
        ]
    else:
        return [ModelWeights(a) for a in bucket.object_versions.all()]


def get_model_weights(model_type: ModelTypeT, version: Optional[str] = None):
    if not version or version == "latest":
        # Get latest weights by default
        return next(w for w in get_available_model_weights(model_type) if w.is_latest)

    # Try to get specified weights
    weights = next(
        w for w in get_available_model_weights(model_type) if w.version_id == version
    )
    if not weights:
        raise ValueError(
            f"Weights version {version} does not exist for model type {model_type}"
        )

    return weights


if __name__ == "__main__":
    weights = get_model_weights("kelp_presence")

    weights.download_if_missing()
