# WienerPytorch
Pytorch implementation of the Wiener image denoising filter. Allow models to be trained through the filter and benefit from CUDA speedup. 

Uses 3D FFT, custom Gaussian windowing,  tunable overlapping block density. 

## Usage
In `test_denoiser.py` replace `dataset_path` with location of noisy images directories. As dataloader is designed for multiple image folders, your directory should follow the structure below: 
```
Base
│
└───dataset_path
│   │
│   └───subfolder1
│       │   file111.png
│       │   file112.png
│   │   
│   └───subfolder2
│       │   file111.png
│       │   file112.png
```

`std` specifies the noisy image standard deviation.

`win_size` specifies the block size for the Wiener filter.

`overlap` specifies the number of overlapping windows in a given block. In other words, selects the sliding window stride. E.g. `win_size = 32` and `overlap = 4`  creates a stride of 8 pixels for the windows.

Denoised images saved to current directory.
