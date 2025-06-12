# CSV2Raster converts a tabular CSV file containing spatial (x, y) and attribute data into a raster format (NetCDF or GeoTIFF).
#
# **Importing and Using the CSV2Raster Class in a Python Program**
#
#         from geoanalytics.conversion import CSV2Raster
#
#         converter = CSV2Raster(inputFile="data.csv", outputFile="output.nc", sep=",", header=None)
#
#         converter.run()
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
    """
    **About this algorithm**

    :**Description**: CSV2Raster converts a structured CSV file with spatial coordinates (x, y) and attributes into raster format (NetCDF or GeoTIFF). It utilizes GDAL for rasterization and CDO for NetCDF operations.

    :**Parameters**:    - **inputFile** (*str*): Path to the input CSV file.
                        - **outputFile** (*str*): Desired name of the output raster file. Default is 'output.nc'.
                        - **sep** (*str*): Delimiter used in the CSV file. Default is space `" "`.
                        - **header** (*int* or *None*): Whether to consider the CSV file's first row as header. Default is `None`.
                        - **dataframe** (*pd.DataFrame* or *str*): Optional DataFrame input (used if `inputFile` is not provided).

    :**Attributes**:    - **inputFile** (*str*): Path of the input CSV file.
                        - **inputfileSep** (*str*): CSV field separator.
                        - **header** (*int* or *None*): Header configuration for the CSV file.
                        - **dataFrame** (*pd.DataFrame*): Loaded and processed data for conversion.
                        - **tempOut** (*str*): Temporary output file name to manage formats.
                        - **outputFile** (*str*): Final raster output file (default `output.nc`).

    **Execution methods**

    .. code-block:: python

            from geoanalytics.conversion import CSV2Raster

            converter = CSV2Raster(inputFile='input.csv', outputFile='converted.tif', sep=',')

            converter.run()

    **Credits**

    This implementation was created and revised under the guidance of Professor Rage Uday Kiran.
    """

    def __init__(self, inputFile='', outputFile='output.nc', sep=" ", header = None, dataframe=''):
        """
        Constructor to initialize the CSV2Raster converter.

        :param input_file: Path to the CSV file.
        :param output_file: Target output filename (NetCDF or GeoTIFF).
        :param sep: Delimiter used in the CSV file.
        :param dataframe: Optional Pandas DataFrame if data is already loaded.
        """

        self.inputFile = inputFile
        self.inputfileSep = sep
        self.header = header
        self.tempOut = outputFile
        self.outputFile = 'output.nc'
        self.dataFrame = dataframe

    def run(self):
        """
        Converts the CSV data to raster format using GDAL and CDO.

        1. Load CSV data into a Pandas DataFrame (if not already provided).
        2. Sort the data by spatial coordinates ('y', then 'x').
        3. Convert each feature column into a temporary NetCDF file using GDAL.
        4. Rename the default variable 'Band1' to actual column names using `ncrename`.
        5. Concatenate all NetCDF files into a single file using `cdo cat`.
        6. Convert to final output format: NetCDF or GeoTIFF.
        7. Clean up temporary files.
        """

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
