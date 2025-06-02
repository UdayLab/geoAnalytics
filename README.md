![PyPI](https://img.shields.io/pypi/v/geoAnalytics)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/geoAnalytics)
[![GitHub license](https://img.shields.io/github/license/UdayLab/geoAnalytics)](https://github.com/UdayLab/geoAnalytics/blob/main/LICENSE)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/geoAnalytics)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/geoAnalytics)
![PyPI - Status](https://img.shields.io/pypi/status/geoAnalytics)
[![GitHub issues](https://img.shields.io/github/issues/UdayLab/geoAnalytics)](https://github.com/UdayLab/geoAnalytics/issues)
[![GitHub forks](https://img.shields.io/github/forks/UdayLab/geoAnalytics)](https://github.com/UdayLab/geoAnalytics/network)
[![GitHub stars](https://img.shields.io/github/stars/UdayLab/geoAnalytics)](https://github.com/UdayLab/geoAnalytics/stargazers)
[![Documentation Status](https://readthedocs.org/projects/geoanalytics/badge/?version=latest)](https://geoanalytics.readthedocs.io/en/latest/?badge=latest)
[![PyPI Downloads](https://static.pepy.tech/badge/geoanalytics)](https://pepy.tech/projects/geoanalytics)
[![PyPI Downloads](https://static.pepy.tech/badge/geoanalytics/month)](https://pepy.tech/projects/geoanalytics)
[![PyPI Downloads](https://static.pepy.tech/badge/geoanalytics/week)](https://pepy.tech/projects/geoanalytics)
[![pages-build-deployment](https://github.com/UdayLab/geoAnalytics/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/UdayLab/geoAnalytics/actions/workflows/pages/pages-build-deployment)




[Click here for more information](https://pepy.tech/project/geoAnalytics)


***

# Table of Contents

- [Introduction](#introduction)
- [Development process](#process-flow-chart)
- [Inputs and outputs of a knowledge discovery algorithm](#inputs-and-outputs-of-an-algorithm-in-pami)
- [Recent updates](#recent-updates)
- [Features](#features)
- [Installing GDAL](#Installing-GDAL-Package)
- [Maintenance](#Maintenance)
- [Try your first geoAnalytics program](#try-your-first-geoAnalytics-program)
- [Reading Material](#Reading-Material)
- [License](#License)
- [Documentation](#Documentation) 
- [Getting Help](#Getting-Help)
- [Discussion and Development](#Discussion-and-Development)
- [Contributors](#Contributors)
- [Tutorials](#tutorials)
  - [Raster conversion](#Raster-conversion)
  - [Imputation](#0-association-rule-mining)
  - [Normalization](#1-pattern-mining-in-binary-transactional-databases)
  - [Classification](#2-pattern-mining-in-binary-temporal-databases)
  - [Clustering](#3-mining-patterns-from-binary-geo-referenced-or-spatiotemporal-databases)
  - [Pattern Mining](#Pattern-mining)
  - [Score Calculation](#4-mining-patterns-from-utility-or-non-binary-databases)
- [Real-World Case Studies](#real-world-case-studies)


***
# Introduction

geoAnalytics is an open-source Python-based Machine Learning library developed to discover various forms of 
useful information hidden in the raster data. The algorithms provided in this library cover a wide-spectrum 
of machine learning tasks, such as imputation, image fusion, clustering, classification, one class 
classification, and pattern mining. This library being platform independent can run any operating system. 
Useful links to utilize the services of this library were provided below:

1. Youtube tutorial  

2. Tutorials (Notebooks)  
   
3. User manual 

4. Coders manual  

5. Code documentation 

6. Datasets  

7. [Discussions](https://github.com/UdayLab/geoAnalytics/discussions)

8. [Report issues](https://github.com/UdayLab/geoAnalytics/issues)

***
# Flow Chart of Developing Algorithms in geoAnalytics

![geoAnalytics production process](./images/geoAnlaytics.png?raw=true)

<!--- ![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true) ---> 
***
# Inputs and Outputs of an Algorithm in geoAnalytics

![Inputs and Outputs](./images/inputsOutputsOfAnAlgorithm?raw=true)
***
# Recent Updates

- **Version 2025.04.29:** 
In this latest version, the following updates have been made:
  - Included one new algorithms, **PrefixSpan**, for Sequential Pattern.
  - Optimized the following pattern mining algorithms: **PFPGrowth, PFECLAT, GPFgrowth and PPF_DFS**.
  - Test cases are implemented for the following algorithms, **Contiguous Frequent patterns, Correlated Frequent Patterns, Coverage Frequent Patterns, Fuzzy Correlated Frequent Patterns, Fuzzy Frequent Patterns, Fuzzy Georeferenced Patterns, Georeferenced Frequent Patterns, Periodic Frequent Patterns, Partial Periodic Frequent Patterns, HighUtility Frequent Patterns, HighUtility Patterns, HighUtility Georeferenced Frequent Patterns, Frequent Patterns, Multiple Minimum Frequent Patterns, Periodic Frequent Patterns, Recurring Patterns, Sequential Patterns, Uncertain Frequent Patterns, Weighted Uncertain Frequent Patterns**.
  - The algorithms mentioned below are automatically tested, **Frequent Patterns, Correlated Frequent Patterns, Contiguous Frequent patterns, Coverage Frequent Patterns, Recurring Patterns, Sequential Patterns**.

Total number of algorithms: 89

***
# Features

- ✅ Tested to the best of our possibility
- 🔋 Highly optimized to our best effort, light-weight, and energy-efficient
- 👀 Proper code documentation
- 🍼 Sample examples of using various algorithms at [./notebooks](https://github.com/UdayLab/PAMI/tree/main/notebooks) folder
- 🤖 Works with AI libraries such as TensorFlow, PyTorch, and sklearn. 
- ⚡️ Supports Cuda 
- 🖥️ Operating System Independence
- 🔬 Knowledge discovery in static data and streams
- 🐎 Snappy
- 🐻 Ease of use

***
# Installing GDAL Package
`GDAL` is an important toolkit in our library. It is for converting the raster data in any format into a human readable text or CSV format.
We have present the methods to install this toolkit using Conda environment on a machine running Ubuntu operating system.


    sudo apt-get update && sudo apt upgrade -y && sudo apt autoremove

    sudo apt-get install -y cdo nco gdal-bin libgdal-dev
    
    
    pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`


    python -m pip install --upgrade pip setuptools wheel
    python -m pip install --upgrade gdal 

If the above two commands have failed to install gdal, then execute the following commands:

    conda install -c conda-forge libgdal
    conda install -c conda-forge gdal
    conda install tiledb=2.2
    conda install poppler

Once the above commands were executed, check the version information by typing the following command on the `terminal`:

    ogrinfo --version

***
# Maintenance

  __Installation__


         pip install geoAnalytics


  __Upgradation__

  
        pip install --upgrade geoAnalytics
  

  __Uninstallation__

  
        pip uninstall geoAnalytics 
       

  __Information__ 


        pip show geoAnalytics

***
# *Try your first geoAnalytics program*

```shell
$ python
```

```python
# first import pami 
from PAMI.frequentPattern.basic import FPGrowth as alg
fileURL = "https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_T10I4D100K.csv"
minSup=300
obj = alg.FPGrowth(iFile=fileURL, minSup=minSup, sep='\t')
#obj.mine()  #deprecated
obj.mine()
obj.save('frequentPatternsAtMinSupCount300.txt')
frequentPatternsDF= obj.getPatternsAsDataFrame()
print('Total No of patterns: ' + str(len(frequentPatternsDF))) #print the total number of patterns
print('Runtime: ' + str(obj.getRuntime())) #measure the runtime
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

```
Output:
Frequent patterns were generated successfully using frequentPatternGrowth algorithm
Total No of patterns: 4540
Runtime: 8.749667644500732
Memory (RSS): 522911744
Memory (USS): 475353088
```
***
# License

[![GitHub license](https://img.shields.io/github/license/UdayLab/PAMI)](https://github.com/UdayLab/PAMI/blob/main/LICENSE)
***

# Documentation

The official documentation is hosted on [geoAnalytics](https://geoanalytics.readthedocs.io/en/stable/).
***
# Getting Help

For any queries, the best place to go to is Github Issues [Github Issues](https://github.com/orgs/UdayLab/discussions/categories/q-a).

***
# Discussion and Development

In our GitHub repository, the primary platform for discussing development-related matters is the university lab. We encourage our team members and contributors to utilize this platform for a wide range of discussions, including bug reports, feature requests, design decisions, and implementation details.

***
# Contribution to geoAnalytics

We invite and encourage all community members to contribute, report bugs, fix bugs, enhance documentation, propose improvements, and share their creative ideas.



# Real World Case Studies

1. Air pollution analytics <a target="_blank" href="https://colab.research.google.com/github/udayLab/PAMI/blob/main/notebooks/airPollutionAnalytics.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


[Go to Top](#table-of-contents)



# geoAnalytics

## Installation using Anaconda.

1. Install and set up Anaconda. URL:   https://linuxize.com/post/how-to-install-anaconda-on-centos-7
2. Create a virtual environment using conda. E.g., coda create --name geoAnalytics
3. Enter into virtual environment.  E.g., conda activate geoAnalytics
4. Install python.   E.g., conda install python
5. Install pycharm from the website
6. Open Pycharm and using VCS download the latest copy of geoAnalytics from GitHub
7. In the pycharm, add geoAnalytics as the interpreter
8. Open the terminal in pycharm, and execute the following command

          pip install mplcursors matplotlib sklearn pandas
          
