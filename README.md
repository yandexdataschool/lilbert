# lilbert

The main code of the project is in lilbert directory. The zoo folder is for collecting intermediate experiments and results that are currently too incomplete to be put to the main directory. Let's try to make the code for the experiments as clean and readable as possible so that other people can understand and reproduce them with minimal effort. Following PEP-8 guidelines and using python scripts (instead of creating large code blocks in jupyter) are greatly encouraged.

For each experiment, there should be a separate directory.

For reproducibility reasons, we all shall use a virtual environment with the same setup: same version of python and all of the packages.
The latest stable version of python is Python 3.7.2, so make sure you download this version from python.org: https://www.python.org/downloads/release/python-372/

After this, for setting up a virtual environment and activating it, run in your shell:
```
virtualenv --python=<path/to/python3.7.2> lilbert_env
source lilbert_env/bin/activate
```
After activating the environment, install all the packages using
```
pip install -r requirements.txt
```
