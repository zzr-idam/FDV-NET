# FDV-NET
FD-Vision Mamba for Endoscopic Exposure Correction

## Installation

- `pip install causal-conv1d>=1.1.0`: an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
- `pip install mamba-ssm`: the core Mamba package.

It can also be built from source with `pip install .` from this repository.

If `pip` complains about PyTorch versions, try passing `--no-build-isolation` to `pip`.

Other requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+

# Dataset

https://pan.baidu.com/s/1y3jo8eSsZ9w03Fw6JkManA?pwd=0nj9 

0nj9 


## Citation

```
@article{zheng2024fd,
  title={FD-Vision Mamba for Endoscopic Exposure Correction},
  author={Zheng, Zhuoran and Zhang, Jun},
  journal={arXiv preprint arXiv:2402.06378},
  year={2024}
}
```



