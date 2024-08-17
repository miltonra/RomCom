# RomComma

**Reduced Order Modelling via Global Sensitivity Analysis using Gaussian Process Regression by Robert A. Milton of The University of Sheffield**

## CUDA Prerequisites for GPU usage
GPU usage requires a `CUDA` setup appropriate to the version of TensorFlow installed. 
Please consult you system administrator or the [TensorFlow Installation Guide](https://www.tensorflow.org/install) for further details.

Under `Windows 11`, GPU usage was abandoned after TensorFlow 2.10 (which requires `Python 3.10`). 
If you wish to use GPU acceleration under `Windows 11` it is recommended that you follow the first 5 Sections of 
[Install TensorFlow GPU on Windows](https://www.lavivienpost.com/install-tensorflow-gpu-on-windows-complete-guide/)
to install `CUDA Toolkit 11.8` and `cuDNN 8.6`.
After this, simply install dependencies according to [pyproject.toml](https://github.com/C-O-M-M-A/rom-comma/blob/main/pyproject.toml) 
for `Python 3.10`.

## Installation
Once your CUDA prerequisites are satisfied, it is recommended that you create and activate a (`virtualenv` or `conda`) environment 
with the desired `Python (3.10 or 3.11)` then
```
git clone https://github.com/C-O-M-M-A/rom-comma.git
cd rom-comma
pip install .
python installation_test.py
```
This should install the correct runtime dependencies. These are exhaustively documented in 
[pyproject.toml](https://github.com/C-O-M-M-A/rom-comma/blob/main/pyproject.toml).

## Documentation
All documentation for the `romcomma` package is published in the [RomComma user guide](https://c-o-m-m-a.github.io/rom-comma/).
