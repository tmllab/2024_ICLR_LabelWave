<h2 align="center">Early Stopping Against Label Noise Without Validation Data</h2>
<p align="center"><b>ICLR 2024 Poster</b> | <a href="https://openreview.net/pdf?id=CMzF2aOfqp">[Paper]</a> | <a href="https://github.com/tmllab/2024_ICLR_LabelWave">[Code]</a> </p>
<p align="center"> <a href="https://suqinyuan.github.io">Suqin Yuan</a>,  <a href="https://lfeng1995.github.io">Lei Feng</a>, <a href="https://tongliang-liu.github.io">Tongliang Liu</a> </p>

### TL;DR
The Label Wave method enables early stopping by analyzing training dynamics in learning with noisy labels, eliminating the need for separate validation data.

### BibTeX
```bibtex
@inproceedings{
yuan2024early,
title={Early Stopping Against Label Noise Without Validation Data},
author={Suqin Yuan and Lei Feng and Tongliang Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
```

### Dependencies
We implement our methods by PyTorch on NVIDIA RTX 3090&4090 GPU. The key environments is as bellow:
- [PyTorch](https://PyTorch.org/), version = 1.11.0
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 11.3

### Experiments
You should put the [CIFAR datasets](https://www.cs.toronto.edu/~kriz/cifar.html) in the folder `.\cifar-10` and `.\cifar-100` when you have downloaded them.


Here is a training example: 
```bash
python3 main.py
```

Contact: Suqin Yuan (suqinyuan.cs@gmail.com).
 
