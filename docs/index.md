**[Home](index.html) |  [Real-world Examples](examples.html)**

# Tutorial on geoAnalytics Python Package

## Topic 1: Setting up the computing environment

Our geoAnalytics package uses  PostGres and PostGIS for storing raster data. Furthermore, geoAnalytics package uses SPARK for running distributed algorithms. Thus, it is important for
the administrator to setup the necessary computing environment.  
  
###1.1. Pre-requisite

- [Setting up PostGres and PostGIS in Ubuntu](postGres.html)
- [Setting up of Hadoop and Spark](https://phoenixnap.com/kb/install-spark-on-ubuntu)  [Spark environment is not mandatory]
- [Install GDAL and other library](gdal.html)

###1.2. Installation

        pip install geoAnalytics

## Topic 2:  Repository connection

Execute the below commands to establish a connection to the repository. Please note that the below commands have to be executed only once to establish connection for the next time.

### 2.1. Establishing connection to the repository
    from osgeo import gdal
    from geoAnalytics import repository as repo
    repo.connect(repositoryName='repositoryName',hostIP='ipaddress',user='userName', password='passwd')
    # Re-execute the above step if there exists any typo 

### 2.2. Testing the repository connection

    repo.testConnection()

Please contact the system administrator if you face any issues after executing the above command.

## Topic 3: Repository maintenance

### 3.1. Creation of a new repository

    repo.create(repositoryName='repoName',totalBands=numberOfBands)

### 3.2. Cloning a repository

    repo.clone(repositoryName='sourceRepositoryName', cloneRepositoryName='newRepositoryName')

### 3.3. Deleting a band in a repository

    repo.deleteBand(repositoryName='clone_kaguya_MI_test', bandNumber='1')

### 3.4. Deleting a repository

    repo.delete(repositoryName='clone_kaguya_MI_test')

### 3.5. Size of a repository

    repo.getSize(repositoryName='sourceRepositoryName')

## Topic 4: Reading a repository


## Topic 5: Data analytics

### 5.1. Clustering algorithms


