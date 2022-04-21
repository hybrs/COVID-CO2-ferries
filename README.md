[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6473158.svg)](https://doi.org/10.5281/zenodo.6473158)

### Paper Title:  "The footprint of COVID-19 on ferry CO2 emissions in Europe"

### Authors    :   G. Mannarini, M.L. Salinas, L.Carelli, A. Fassò

<hr>

<a id='toc' name='toc'></a>
# Table of Contents

 - [About](#about)
 
 - [Usage](#usage)

<a id='about' name='about'></a>
## About [[to ToC]](#toc)

This repository contains auxiliary material useful to replicate the paper results:

  - `README.md`,<br>
  this file

  - `notebook.ipynb`,<br>
  jupyter notebook that reproduces the paper results performing a 10-fold cross-validation on the same models

  - `su-venv.yml`,<br>
  conda virtual-env to be imported in order to run the provided notebook

  - `cv_routines.py`,<br>
  core cross-validation python code

<a id='usage' name='usage'></a>
## Usage [[to ToC]](#toc)
  
  1. install [Anaconda](https://www.anaconda.com/products/distribution)
  
  2. import conda virtual environment from sustainability-venv.yml with
  ```bash
  # WAITING TIME ~ 20min
  conda env create -f su-venv.yml
  ```
  
  3. launch and run the provided notebook inside su-venv environment with
  ```bash
  conda activate su-venv  
  jupyter-notebook
  ```


---
**Copyright (c) Fondazione CMCC, 2022**

This work was supported by the European Regional Development Fund through the Italy-Croatia Interreg programme, project GUTTA, grant number 10043587.

---
