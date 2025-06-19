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
This installs and tests all the correct runtime dependencies, as listed in 
[pyproject.toml](https://github.com/miltonra/RomCom/blob/main/pyproject.toml).

## Documentation
All documentation for the ``RomCom`` library is published online in the 
[RomCom User Guide](https://miltonra.github.io/RomCom/).
The same documentation is available locally in 
[docs/index.html](https://github.com/miltonra/RomCom/blob/main/docs/index.html).

## Rebuilding Documentation
In the unlikely event that you need to rebuild the local documentation, navigate to 
[docs/sphinx/](https://github.com/miltonra/RomCom/blob/main/docs/sphinx) and run::

    make tidy
Win11 users who have not installed the GNU ``make`` utlility should instead run::

    .\make.bat tidy

Either command runs ``make clean``, tidies the content some content in the resulting HTML files, which is slow.

A much quicker way to build flawed documentation is to run ``make clean`` 
which deletes all sphinx artefacts, then runs ``make dirty``.

The ``make dirty [OR quick OR html]`` command just runs the Sphinx build, which is fast but will produce flawed documentation.

As usual, help is available by running ``make help`` or ``make -h``, or just ``make``.

Examine the makefiles  in 
[docs/sphinx/](https://github.com/miltonra/RomCom/blob/main/docs/sphinx) 
for further details. The ``clean`` and ``tidy`` aspects of ``make`` are provided by 
[docs/sphinx/source/utilities.py](https://github.com/miltonra/RomCom/blob/main/docs/sphinx/source/utilities.py). 
