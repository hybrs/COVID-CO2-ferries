[![DOI:10.3390](http://img.shields.io/badge/DOI-10.3390/su14095287-blue.svg)](https://doi.org/10.3390/su14095287)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6473158.svg)](https://doi.org/10.5281/zenodo.6473158)

![Ubuntu](https://img.shields.io/badge/Linux-passed-green)
![macOs](https://img.shields.io/badge/MacOs-passed-green)
![Windows](https://img.shields.io/badge/Windows-passed-green)

![Conda](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)
![jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)

# How COVID-19 affected GHG emissions of ferries in Europe
<table>
<thead>
  <tr>
    <td><a href="https://orcid.org/0000-0001-9205-7765" target='_blank'><img width=15 src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg"></a> G. Mannarini<sup>1</sup></td>
    <td><a href="https://orcid.org/0000-0002-4045-4790" target='_blank'><img width=15 src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg"></a> M.L. Salinas<sup>1</sup></td>
    <td><a href="https://orcid.org/0000-0003-4259-3505" target='_blank'><img width=15 src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg"></a> L. Carelli<sup>1</sup></td>
    <td><a href="https://orcid.org/0000-0001-5132-9488" target='_blank'><img width=15 src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg"></a>   A. Fassò<sup>2</sup></td>
  </tr>
</thead>
</table>

---

<a id='toc' name='toc'></a>
# Table of Contents

 - [About](#about)
 
 - [Usage](#usage)

 - [References](#ref)

---

<a id='about' name='about'></a>
## About [[to ToC]](#toc)

This repository contains auxiliary material useful to replicate the results of the *"How COVID-19 affected GHG emissions of ferries in Europe"*
, which is currently under review at [Sustainability](https://www.mdpi.com/journal/sustainability). Once published, you may want to **cite it.**

The table below describes the content of the repository.

| filename       | description                                                                                                 |
|----------------|-------------------------------------------------------------------------------------------------------------|
| `README.md`     | this file                                                                                                   |
| `notebook.ipynb` | jupyter notebook that reproduces the paper results performing a 10-fold cross-validation on the same models |
| `cv_routines.py` | core cross-validation python code                                                                           |
| `su-venv.yml`    | conda virtual-env to be imported in order to run the provided notebook                                      |
| `LICENSE`    | BSD 2-Clause License                               |

<a id='usage' name='usage'></a>
## Usage [[to ToC]](#toc)
  
  1. install [Anaconda](https://www.anaconda.com/products/distribution)
  
  2. import conda virtual environment from sustainability-venv.yml with
  ```bash
  # WAITING TIME ~ 10min
  conda env create -f su-venv.yml
  ```
  
  3. launch and run the provided notebook inside su-venv environment with
  ```bash
  conda activate su-venv  
  jupyter-notebook
  ```

---

<a id='ref' name='ref'></a>
## References [[to ToC]](#toc)

[![DOI:10.3390](http://img.shields.io/badge/DOI-10.3390/su14095287-blue.svg)](https://doi.org/10.3390/su14095287)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6473158.svg)](https://doi.org/10.5281/zenodo.6473158)


Please, when using this code and the data available on Zenodo, cite the references listed below according to your needs:

```
@dataset{mannarini_gianandrea_2022_6473158,
  author       = {Mannarini, Gianandrea and
                  Salinas, Mario and
                  Carelli, Lorenzo and
                  Fassò, Alessandro},
  title        = {COVID-CO2-ferries},
  month        = {April},
  year         = {2022},
  note         = {(c) Fondazione CMCC, 2022  ERDF through Italy-Croatia Interreg Programme: 10043587 (GUTTA)},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6473158},
  url          = {https://doi.org/10.5281/zenodo.6473158}
}



@article{su14095287,
  author         = {Mannarini, Gianandrea and Salinas, Mario Leonardo and Carelli, Lorenzo and Fassò, Alessandro},
  title          = {How COVID-19 Affected GHG Emissions of Ferries in Europe},
  journal        = {Sustainability},
  volume         = {14},
  year           = {2022},
  number         = {9},
  article-number = {5287},
  note           = {(c) Fondazione CMCC, 2022  ERDF through Italy-Croatia Interreg Programme: 10043587 (GUTTA)}
  url            = {https://www.mdpi.com/2071-1050/14/9/5287},
  issn           = {2071-1050},
  doi            = {10.3390/su14095287}
}

```

---
**Copyright (c) Fondazione CMCC, 2022**

*ERDF through Italy-Croatia Interreg Programme: 10043587 (GUTTA)*

---

<sup>1</sup> [CMCC Foundation](http://www.cmcc.it)

<sup>2</sup> [Università degli studi di Bergamo](http://www.unibg.it)
