**[Home](index.html) |  [Real-world Examples](examples.html)**
 
# Installing GDAL and related libraries

- Open terminal and execute below commands as the administrator

      sudo apt-get install -y cdo nco      #necessary for geoAnalytics package.
      sudo add-apt-repository ppa:ubuntugis/ppa
      sudo apt-get update
      sudo apt-get install gdal-bin
      sudo apt-get install libgdal-dev

- Open .bashrc file and add the below two lines

      export CPLUS_INCLUDE_PATH=/usr/include/gdal
      export C_INCLUDE_PATH=/usr/include/gdal

- Save the file. Compile the .bashrc file by executing the following command

      source .bashrc

- Execute the following command on the terminal.

      pip install gdal.
