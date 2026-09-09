# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]
REQUIREMENTS += [
    "CLIP @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33#egg=CLIP"
]

setup(
    name="fastsam",
    version="0.1.1",
    install_requires=REQUIREMENTS,
    packages=find_packages(include=("fastsam", "fastsam.*", "ultralytics", "ultralytics.*")),
    python_requires=">=3.9",
    url="https://github.com/CASIA-LMC-Lab/FastSAM",
)
