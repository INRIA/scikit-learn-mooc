# Local install instructions

The course uses Python 3 and some data analysis packages such as Numpy, Pandas,
scikit-learn, and matplotlib.

## Install Miniconda

**This step is only necessary if you don't have conda installed already**:

- download the Miniconda installer for your operating system (Windows, MacOSX
  or Linux) [here](https://docs.conda.io/en/latest/miniconda.html)
- run the installer following the instructions
  [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)
  depending on your operating system.

## Create conda environment

```sh
# Clone this repo
git clone https://github.com/INRIA/scikit-learn-mooc
cd scikit-learn-mooc
# Create a conda environment with the required packages for this tutorial:
conda env create -f environment.yml
```

## Check your install 

To make sure you have all the necessary packages installed, we **strongly
recommend** you to execute the `check_env.py` script located at the root of
this repository:

```sh
# Activate your conda environment
conda activate scikit-learn-course
python check_env.py
```

Make sure that there is no `FAIL` in the output when running the `check_env.py`
script, i.e. that its output looks similar to this:

```
Using python in /home/lesteve/miniconda3/envs/scikit-learn-course
3.9.1 | packaged by conda-forge | (default, Jan 10 2021, 02:55:42)
[GCC 9.3.0]

[ OK ] numpy version 1.19.5
[ OK ] scipy version 1.6.0
[ OK ] matplotlib version 3.3.3
[ OK ] sklearn version 0.24.0
[ OK ] pandas version 1.2.0
[ OK ] seaborn version 0.11.1
[ OK ] notebook version 6.2.0
[ OK ] plotly version 4.14.3
```

## Run Jupyter notebooks locally

```sh
# Activate your conda environment
conda activate scikit-learn-course
jupyter notebook index.md
```

`index.md` is an index file helping to navigate the notebooks.
All the Jupyter notebooks are located in the `notebooks` folder.

