# RomCom

**Reduced Order Modelling via GSA/GPR, by Robert A. Milton and Solomon F. Brown**

## CUDA Prerequisites for GPU usage
GPU usage requires a ``CUDA`` setup appropriate to the version of PyTorch installed.
Please consult you system administrator or [PyTorch Get Started](https://pytorch.org/get-started/locally/) for further details.

## Python Version
Python 3.12 is currently supported. 
Other Python versions may work, but are untested. 
Any issues are most likely raised by PyTorch and its dependencies NumPy and pandas.

## Installation
Once your CUDA prerequisites are satisfied, it is recommended that you create and activate a (``virtualenv`` or ``conda``) environment 
with the desired ``python`` then
```
git clone https://github.com/miltonra/RomCom.git
cd RomCom
pip install .
python installation_test.py
```
This installs and tests all the correct runtime dependencies, as listed in [pyproject.toml](https://github.com/miltonra/RomCom/blob/main/pyproject.toml).

## Documentation
All documentation for the ``RomCom`` library is published in the [RomCom User Guide](https://miltonra.github.io/RomCom/).
