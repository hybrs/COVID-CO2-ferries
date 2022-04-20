### Paper Title:  "The footprint of COVID-19 on ferry CO2 emissions in Europe"

### Authors    :   G. Mannarini, M.L. Salinas, L.Carelli, A. Fassò

<hr>

#### Copyright (c) 2022 - CMCC

<hr>

# About

This repository contains auxiliary material useful to replicate the paper results:

  - `README.md`,<br>
  this file

  - `notebook.ipynb`,<br>
  jupyter notebook that reproduces the paper results performing a 10-fold cross-validation on the same models

  - `su-venv.yml`,<br>
  conda virtual-env to be imported in order to run the provided notebook

  - `cv_routines.py`,<br>
  core cross-validation python code


<hr>

# Usage:
  
  1. install [Anaconda](https://www.anaconda.com/products/distribution)
  
  2. import conda virtual environment from sustainability-venv.yml with

    WAITING TIME ~ 20min
    > conda env create -f su-venv.yml
  
  3. launch and run the provided notebook inside su-venv environment with
  
    > conda activate su-venv  
    > jupyter-notebook

