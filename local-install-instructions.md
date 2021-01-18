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

## Run Jupyter notebooks locally

```sh
# Activate your conda environment
conda activate scikit-learn-course
jupyter notebook notebooks/index.md
```

`notebooks/index.md` is an index file helping to navigate the notebooks.
All the Jupyter notebooks are located in the `notebooks` folder.

