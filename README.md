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

2. Tutorials
   
3. User manual 

4. Coders manual  

5. Code documentation 

6. Datasets  

7. [Discussions](https://github.com/UdayLab/geoanalytics/discussions)

8. [Report issues](https://github.com/UdayLab/geoanalytics/issues)

***
# Flow Chart of Developing Algorithms in geoAnalytics

![geoAnalytics production process](https://github.com/UdayLab/geoanalytics/blob/main/images/geoAnalyticsLibrary.png?raw=true)


<!--- ![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true) ---> 
***
# Inputs and Outputs of an Algorithm in geoAnalytics

![Inputs and Outputs](https://github.com/UdayLab/geoanalytics/blob/main/images/inputsOutputsOfAnAlgorithm.png?raw=true)

***
# Recent Updates

- **Version 2025.04.29:** 
In this latest version, the following updates have been made:
  - Included thirteen new algorithms in imputation, **BackwardFill**, **ForwardFill**, **HotDeck**, **Interpolation**, **KNNImputation**, **MatrixFactorization**, **MeanImputation**, **MedianImputation**, **MICEImputation**, **ModeImputation**, **MultipleImputation**, **NumberImputation**, and **SoftImputation** for filling nan values.
  - Test cases are implemented using geoanalytics package.
Total number of algorithms: 30+
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
# first import geoanalytics 
from geoanalytics.clustering import KMeans as alg
import pandas as pd
df = pd.read_csv('Moon.csv',header=None,sep=',')
obj = alg.KMeans(dataframe=df)
obj.elbowMethod()
obj.clustering(k=4,max_iter=100)
obj.save('KMeansLabels.csv')
```

```
Output:
Total Execution time of proposed Algorithm: 7.29867959022522
Memory (USS) of proposed Algorithm in KB: 3005180.0
Memory (RSS) of proposed Algorithm in KB: 3025648.0

              x	             y	      labels
0	1061317.265	-485173.607	2
1	1061332.071	-485173.607	2
2	1061346.877	-485173.607	2
3	1061361.684	-485173.607	1
4	1061376.490	-485173.607	1
...	    ...	             ...	...
4194299	1091566.583	-515482.151	2
4194300	1091581.390	-515482.151	2
4194301	1091596.196	-515482.151	2
4194302	1091611.002	-515482.151	2
4194303	1091625.809	-515482.151	2

array([[2145.99705143, 3838.28679175, 4174.26050222, 4214.72552938,
        4362.65581646, 4291.2789215 , 4508.31819976, 5545.56035569,
        6874.39746827],
       [1904.93217264, 3435.9266567 , 3754.57334822, 3814.21158   ,
        3957.99354212, 3906.56010996, 4104.85927441, 5081.01403926,
        6344.78487709],
       [2007.81883172, 3616.11456616, 3945.08473418, 3996.36984105,
        4144.25548759, 4082.02786188, 4290.37009057, 5300.17673097,
        6597.33840088],
       [2426.77124575, 4271.09574647, 4574.37251878, 4570.90689388,
        4707.65309669, 4610.27659044, 4838.39224733, 5938.90730475,
        7321.7573098 ]])
        
Labels saved to: KMeansLabels.csv
```
***
# License

[![GitHub license](https://img.shields.io/github/license/UdayLab/PAMI)](https://github.com/UdayLab/PAMI/blob/main/LICENSE)
***

# Documentation

The official documentation is hosted on [geoAnalytics](https://geoanalytics.readthedocs.io/en/latest/).
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

1. Lunar data analytics <a target="_blank" href="https://colab.research.google.com/github/udayLab/PAMI/blob/main/notebooks/airPollutionAnalytics.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


[Go to Top](#table-of-contents)

## Installation using Anaconda (geoanalytics package).

1. Install and set up Anaconda. URL:   https://linuxize.com/post/how-to-install-anaconda-on-centos-7
2. Create a virtual environment using conda. E.g., coda create --name geoAnalytics
3. Enter into virtual environment.  E.g., conda activate geoAnalytics
4. Install python.   E.g., conda install python
5. Install pycharm from the website
6. Open Pycharm and using VCS download the latest copy of geoAnalytics from GitHub
7. In the pycharm, add geoAnalytics as the interpreter
8. Open the terminal in pycharm, and execute the following command

          pip install mplcursors matplotlib sklearn pandas
          
