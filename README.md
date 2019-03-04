# lilbert

For reproducibility reasons, we all shall use a virtual environment with the same setup: same version of python and all of the packages.
The latest stable version of python is Python 3.7.2, so make sure you download this version from python.org.

After this, you can set up a virtual environment and activate it. Run in your shell:
```
virtualenv --python=<path/to/python3.7.2> lilbert_env
source lilbert_env/bin/activate
```
After creating an environment, install all the packages using
```
pip install -r requirements.txt
```
