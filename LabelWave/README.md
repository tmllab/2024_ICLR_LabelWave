# Label Wave

This is the code for the paper:

Early Stopping Against Label Noise Without Validation Data.

ICLR 2024 Poster.
## Dependencies
We implement our methods by PyTorch on NVIDIA RTX 3090&4090 GPU. The environment is as bellow:
- [PyTorch](https://PyTorch.org/), version = 1.11.0
- [Ubuntu20.04](https://ubuntu.com/download)
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 11.3

## Experiments
We verify the effectiveness of the proposed method on noisy datasets. In this repository, we provide the used [datasets](https://www.cs.toronto.edu/~kriz/cifar.html). 
You should put the datasets in the folder ''cifar-10'' and ''cifar-100'' when you have downloaded them.


Here is a training example: 
```bash
python3 main.py
 
