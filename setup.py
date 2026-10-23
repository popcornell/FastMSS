# Copyright 2024-2026 Samuele Cornell and FastMSS contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name="fastmss",
    version="0.1.0",
    description="Fast Multi-Speaker Speech simulation framework",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "soundfile",
        "torch",
        "torchaudio",
        "lhotse",
        "pyroomacoustics",
        "hydra-core",
        "omegaconf",
        "tqdm",
        "urllib3"
    ],
)
