# -*- coding: utf-8 -*-
from setuptools import setup

PACKAGE_NAME = 'hakai-segmentation'
VERSION = '0.3.1'

packages = [
    'hakai_segmentation',
    'hakai_segmentation.data',
    'hakai_segmentation.geotiff_io'
]

package_data = {
    '': ['*']
}

install_requires = [
    'boto3~=1.24',
    'botocore~=1.27',
    'numpy~=1.16',
    'rasterio~=1.2',
    'rich~=12.6',
    'torchvision~=0.11',
    'torch~=1.10',
    'tqdm~=4.62',
    'typer~=0.4',
]

entry_points = {
    'console_scripts': ['kom = hakai_segmentation.cli:cli']
}

setup_kwargs = {
    'name': PACKAGE_NAME,
    'version': VERSION,
    'description': 'Segmentation Tools for Remotely Sensed RPAS Imagery',
    'author': 'Taylor Denouden',
    'author_email': 'taylordenouden@gmail.com',
    'url': 'https://github.com/hakaiInstitute/hakai-segmentation',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '~=3.7,<3.10',
}

setup(**setup_kwargs)
