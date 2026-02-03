# This file is a modification of the open‑source 'qugen' project: https://github.com/QutacQuantum/qugen
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Anonymous contributors
# Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

import platform

from setuptools import setup, find_packages

if platform.system() == 'Linux':
    setup(name='qugen',
          version='0.1',
          author='Quantum Technology & Application Consortium (QUTAC)',
          license='Apache License 2.0',
          packages=find_packages(),
          install_requires=[
              'cma>=3.2.2',
              'colorama>=0.4.5',
              'flax>=0.8.2',
              'jax=0.6.0',
              'jaxlib=0.6.0',
              'matplotlib>=3.5.3',
              'numpy>=1.26.4',
              'optax>=0.2.2',
              'pandas>=2.2.2',
              'PennyLane>=0.39.0',
              'pytest>=7.4.0',
              'scipy>=1.13.0',
              'setuptools>=61.2.0',
              'tqdm>=4.64.1',
              'fire>=0.7',
              'scikit-learn>=1.6.1',
              "torchmetrics[image]>=1.6.2",
              "cuda-selector=0.1.5",
              'opencv-python>=4.11',
              'pytorch-fid>=0.3.0',])
elif platform.system() == 'Darwin':
    setup(name='qugen',
          version='0.1',
          author='Quantum Technology & Application Consortium (QUTAC)',
          license='Apache License 2.0',
          packages=find_packages(),
          install_requires=[
              'cma>=3.2.2',
              'colorama>=0.4.5',
              'flax>=0.8.2',
              'jax==0.6.0',
              'jaxlib==0.6.0',
              'matplotlib>=3.5.3',
              'numpy>=1.26.4',
              'optax>=0.2.2',
              'pandas>=2.2.2',
              'PennyLane>=0.39.0',
              'pytest>=7.4.0',
              'scipy>=1.13.0',
              'setuptools>=61.2.0',
              'tqdm>=4.64.1',
              'fire>=0.7',
              'scikit-learn>=1.6.1',
              "torchmetrics[image]>=1.6.2",
              "cuda-selector==0.1.5",
              'opencv-python>=4.11',
              'pytorch-fid>=0.3.0',])
elif platform.system() == 'Windows':
    setup(name='qugen',
          version='0.1',
          author='Quantum Technology & Application Consortium (QUTAC)',
          license='Apache License 2.0',
          packages=find_packages(),
          install_requires=[
              'cma>=3.2.2',
              'colorama>=0.4.5',
              'flax>=0.8.2',
              'jax=0.6.0',
              'jaxlib=0.6.0',
              'matplotlib>=3.5.3',
              'numpy>=1.26.4',
              'optax>=0.2.2',
              'pandas>=2.2.2',
              'PennyLane>=0.39.0',
              'pytest>=7.4.0',
              'scipy>=1.13.0',
              'setuptools>=61.2.0',
              'tqdm>=4.64.1',
              'fire>=0.7',
              'scikit-learn>=1.6.1',
              "torchmetrics[image]>=1.6.2",
              "cuda-selector=0.1.5",
              'opencv-python>=4.11',
              'pytorch-fid>=0.3.0',
          ])
else:
    raise OSError('Unknown Operating System: {} {}'.format(platform.os.name, platform.system()))
