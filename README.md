# Mamba-in-Speech
# Mamba in Speech: Towards an Alternative to Self-Attention

This repository contains the official implementation of the paper [Mamba in Speech: Towards an Alternative to Self-Attention](https://arxiv.org/abs/2405.12609).

## Overview
For ASR task, we provided espnet related files. 

For Speech Enhancement task, the pipeline is done by Pytorch

## Installation

To use this implementation, you need to install Mamba in your system. You can then install the required packages using pip.

1. Install the `causal-conv1d` package:
    ```bash
    pip install causal-conv1d>=1.2.0
    ```

2. Install the `mamba-ssm` package:
    ```bash
    pip install mamba-ssm
    ```



## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{zhang2024mamba,
  title={Mamba in Speech: Towards an Alternative to Self-Attention},
  author={Zhang, Xiangyu and Zhang, Qiquan and Liu, Hexin and Xiao, Tianyi and Qian, Xinyuan and Ahmed, Beena and Ambikairajah, Eliathamby and Li, Haizhou and Epps, Julien},
  journal={arXiv preprint arXiv:2405.12609},
  year={2024}
}

