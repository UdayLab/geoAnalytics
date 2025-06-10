# csv2Parquet converts the input CSV file to a data frame, which is then transformed into a Parquet file.
#
# **Importing this algorithm into a python program**
#
#             from PAMI.extras.convert import csvParquet as cp
#
#             obj = cp.CSV2Parquet(sampleDB.csv, output.parquet, sep)
#
#             obj.convert()
#
#             obj.printStats()
#

__copyright__ = """
Copyright (C)  2022 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from osgeo import gdal
import pandas as pd
import os
import random
import subprocess

class CSV2Raster:
    def __init__(self, inputFile='', outputFile='output.nc', sep=" ", header = None, dataframe=''):
        self.inputFile = inputFile
        self.inputfileSep = sep
        self.header = header
        self.tempOut = outputFile
        self.outputFile = 'output.nc'
        self.dataFrame = dataframe

    def run(self):
        if self.inputFile != '':
            self.dataFrame = pd.read_csv(self.inputFile, sep=self.inputfileSep, header=None)

        self.dataFrame = self.dataFrame.sort_values(['y', 'x'], ascending=[True, True])
        dataFrameColumns = self.dataFrame.columns

        randInt = str(random.randint(0, 100000))
        self.dataFrame = self.dataFrame.sort_values(by=['y', 'x'], ascending=[False, True])

        for i in range(2, len(dataFrameColumns)):
            self.dataFrame.to_csv("xyzformat.xyz", index=False, header=None, sep=" ")
            raster = gdal.Translate("temp_" + randInt + "_" + str(dataFrameColumns[i]) + ".nc", "xyzformat.xyz")
            self.dataFrame = self.dataFrame.drop(columns=dataFrameColumns[i])
            buffer = 'ncrename -v Band1,' + str(dataFrameColumns[i]) + " temp_" + randInt + "_" + str(dataFrameColumns[i]) + ".nc"
            print(subprocess.getstatusoutput(buffer))

        buffer = 'cdo -f nc2 cat ' + "temp_" + randInt + "_*.nc " + self.outputFile
        print(subprocess.getstatusoutput(buffer))

        buffer = 'rm ' + "temp_" + randInt + "_*.nc"
        print(subprocess.getstatusoutput(buffer))

        if self.tempOut[-3:] == '.nc':
            os.rename(self.outputFile, self.tempOut)
        elif self.tempOut[-3:] == 'iff' or self.tempOut[-3:] == 'tif':
            buffer = 'gdal_translate -of GTiff output.nc ' + self.tempOut
            print(subprocess.getstatusoutput(buffer))
            buffer = 'rm output.nc'
            print(subprocess.getstatusoutput(buffer))
        else:
            os.rename(self.outputFile, self.tempOut + ".nc")

        os.remove("xyzformat.xyz")
