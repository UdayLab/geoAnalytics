from osgeo import gdal
import pandas as pd
import os
import random
import subprocess

def csv2raster(input_file='', output_file='output.nc', sep=" ", dataframe=''):
    input_file = input_file
    inputfile_sep = sep
    tempOut = output_file
    output_file = 'output.nc'
    dataFrame = dataframe
    if input_file != '':
        dataFrame = pd.read_csv(input_file, 
                                     sep=inputfile_sep, 
                                     header=None, index_col=None)


    dataFrame = dataFrame.sort_values(['y','x'], ascending=[True,True])
    dataFrameColumns = dataFrame.columns

    randInt = str(random.randint(0, 100000))
    dataFrame = dataFrame.sort_values(by=['y', 'x'], ascending = [False,True])

    for i in range(2,len(dataFrameColumns)):
        dataFrame.to_csv("xyzformat.xyz", index=False, header=None, sep=" ")
        raster = gdal.Translate("temp_" + randInt + "_" + str(dataFrameColumns[i]) + ".nc", "xyzformat.xyz")
        dataFrame = dataFrame.drop(columns=dataFrameColumns[i])
        buffer = 'ncrename -v ' + 'Band1,' + str(dataFrameColumns[i]) + " temp_" + randInt + "_" + str(dataFrameColumns[i]) + ".nc" 
        print(subprocess.getstatusoutput(buffer))


    buffer = 'cdo -f nc2 cat ' + "temp_" + randInt + "_*.nc " + output_file
    print(subprocess.getstatusoutput(buffer))

    buffer = 'rm ' + "temp_" + randInt + "_*.nc"
    print(subprocess.getstatusoutput(buffer))
    
    if tempOut[-3:] == '.nc':
        #rename
        os.rename(output_file, tempOut)
    elif tempOut[-3:] == 'iff' or tempOut[-3:] == 'tif':
        buffer = 'gdal_translate -of GTiff output.nc ' + tempOut
        print(subprocess.getstatusoutput(buffer))
        buffer = 'rm output.nc'
        print(subprocess.getstatusoutput(buffer))
    else:
        os.rename(output_file, tempOut + ".nc")


    os.remove("xyzformat.xyz")
