![PyPI](https://img.shields.io/pypi/v/geoAnalytics)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/geoAnalytics)
[![GitHub license](https://img.shields.io/github/license/UdayLab/geoAnalytics)](https://github.com/UdayLab/geoAnalytics/blob/main/LICENSE)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/geoAnalytics)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/geoAnalytics)
![PyPI - Status](https://img.shields.io/pypi/status/geoAnalytics)
[![GitHub issues](https://img.shields.io/github/issues/UdayLab/geoAnalytics)](https://github.com/UdayLab/geoAnalytics/issues)
[![GitHub forks](https://img.shields.io/github/forks/UdayLab/geoAnalytics)](https://github.com/UdayLab/geoAnalytics/network)
[![GitHub stars](https://img.shields.io/github/stars/UdayLab/geoAnalytics)](https://github.com/UdayLab/geoAnalytics/stargazers)
[![PyPI Downloads](https://static.pepy.tech/badge/geoanalytics)](https://pepy.tech/projects/geoanalytics)
[![PyPI Downloads](https://static.pepy.tech/badge/geoanalytics/month)](https://pepy.tech/projects/geoanalytics)
[![PyPI Downloads](https://static.pepy.tech/badge/geoanalytics/week)](https://pepy.tech/projects/geoanalytics)
[![pages-build-deployment](https://github.com/UdayLab/geoAnalytics/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/UdayLab/geoAnalytics/actions/workflows/pages/pages-build-deployment)




[Click here for more information](https://pepy.tech/project/geoAnalytics)


***

# Table of Contents

- [Introduction](#introduction)
- [Development process](#process-flow-chart)
- [Inputs and outputs of a PAMI algorithm](#inputs-and-outputs-of-an-algorithm-in-pami)
- [Recent updates](#recent-updates)
- [Features](#features)
- [Maintenance](#Maintenance)
- [Try your first geoAnalytics program](#try-your-first-PAMI-program)
- [Evaluation](#evaluation)
- [Reading Material](#Reading-Material)
- [License](#License)
- [Documentation](#Documentation)
- [Background](#Background)
- [Getting Help](#Getting-Help)
- [Discussion and Development](#Discussion-and-Development)
- [Contribution to geoAnalytics](#Contribution-to-PAMI)
- [Tutorials](#tutorials)
  - [Imputation](#0-association-rule-mining)
  - [Normalization](#1-pattern-mining-in-binary-transactional-databases)
  - [Classification](#2-pattern-mining-in-binary-temporal-databases)
  - [Clustering](#3-mining-patterns-from-binary-geo-referenced-or-spatiotemporal-databases)
  - [Score Calculation](#4-mining-patterns-from-utility-or-non-binary-databases)
- [Real-World Case Studies](#real-world-case-studies)


***
# Introduction

PAttern MIning (PAMI) is a Python library containing several algorithms to discover user interest-based patterns in a wide-spectrum of datasets across multiple computing platforms. Useful links to utilize the services of this library were provided below:

1. Youtube tutorial ?? https://www.youtube.com/playlist?list=PLKP768gjVJmDer6MajaLbwtfC9ULVuaCZ

2. Tutorials (Notebooks) ?? https://github.com/UdayLab/PAMI/tree/main/notebooks
   
3. User manual ?? https://udaylab.github.io/PAMI/manuals/index.html

4. Coders manual ?? https://udaylab.github.io/PAMI/codersManual/index.html

5. Code documentation ?? https://pami-1.readthedocs.io

6. Datasets   https://u-aizu.ac.jp/~udayrage/datasets.html

7. Discussions on PAMI usage https://github.com/UdayLab/geoAnalytics/discussions

8. Report issues https://github.com/UdayLab/geoAnalytics/issues

***
# Flow Chart of Developing Algorithms in PAMI

![PAMI's production process](./images/pamiDevelopmentSteps.png?raw=true)

<!--- ![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true) ---> 
***
# Inputs and Outputs of an Algorithm in PAMI

![Inputs and Outputs](./images/inputOutputPAMIalgo.png?raw=true)
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

# Evaluation:

1. we compared three different Python libraries such as PAMI, mlxtend and efficient-apriori for Apriori.
2. (Transactional_T10I4D100K.csv)is a transactional database downloaded from PAMI and
used as an input file for all libraries.
3. Minimum support values and seperator are also same.

* The performance of the **Apriori algorithm** is shown in the graphical results below:
1. Comparing the **Patterns Generated** by different Python libraries for the Apriori algorithm:

   <img width="573" alt="Screenshot 2024-04-11 at 13 31 31" src="https://github.com/vanithakattumuri/PAMI/assets/134862983/fd7974bc-ffe2-44dd-82e3-a5306a8a23bd">
   
2. Evaluating the **Runtime** of the Apriori algorithm across different Python libraries:

   <img width="567" alt="Screenshot 2024-04-11 at 13 31 20" src="https://github.com/vanithakattumuri/PAMI/assets/134862983/5d615ae3-dc0d-49ba-a880-4890bb1f11c5">

3. Comparing the **Memory Consumption** of the Apriori algorithm across different Python libraries:

   <img width="570" alt="Screenshot 2024-04-11 at 13 31 08" src="https://github.com/vanithakattumuri/PAMI/assets/134862983/5d5991ca-51ae-442d-9b5e-2d21bbebfedd">

For more information, we have uploaded the evaluation file in two formats:
- One **ipynb** file format, please check it here. [Evaluation File ipynb](https://github.com/UdayLab/PAMI/blob/main/notebooks/Evaluation-neverDelete.ipynb) 
- Two **pdf** file format, check here. [Evaluation File Pdf](https://github.com/UdayLab/PAMI/blob/main/notebooks/evaluation.pdf)

***
# Reading Material

For more examples, refer this YouTube link [YouTube](https://www.youtube.com/playlist?list=PLKP768gjVJmDer6MajaLbwtfC9ULVuaCZ)

***
# License

[![GitHub license](https://img.shields.io/github/license/UdayLab/PAMI)](https://github.com/UdayLab/PAMI/blob/main/LICENSE)
***

# Documentation

The official documentation is hosted on [PAMI](https://pami-1.readthedocs.io).
***

# Background

The idea and motivation to develop PAMI was from [Kitsuregawa Lab](https://www.tkl.iis.u-tokyo.ac.jp/new/resources?lang=en) at the University of Tokyo. Work on ``PAMI`` started at [University of Aizu](https://u-aizu.ac.jp/en/) in 2020 and
has been under active development since then.

***
# Getting Help

For any queries, the best place to go to is Github Issues [GithubIssues](https://github.com/orgs/UdayLab/discussions/categories/q-a).

***
# Discussion and Development

In our GitHub repository, the primary platform for discussing development-related matters is the university lab. We encourage our team members and contributors to utilize this platform for a wide range of discussions, including bug reports, feature requests, design decisions, and implementation details.

***
# Contribution to PAMI

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
          
