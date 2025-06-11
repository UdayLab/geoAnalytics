# Raster2CSV converts raster data (NetCDF or TIFF) into tabular CSV/TSV format using coordinate-band representation.
#
# **Importing and Using the Raster2CSV Class in a Python Program**
#
#         from geoanalytics.conversion import Raster2CSV
#
#         converter = Raster2CSV(inputFile='raster.tif', outputFile='output.csv', startBand=1, endBand=5)
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


import os
import pandas as pd
from geoanalytics.conversion import raster2tsv

class Raster2CSV:
    """
    **About this algorithm**

    :**Description**: Raster2CSV is a converter that reads raster files (e.g., NetCDF, TIFF), extracts specified band values, and writes them to a CSV/TSV file along with spatial coordinates (x, y).

    :**Parameters**:    - **inputFile** (*str*): Path to the raster file.
                        - **outputFile** (*str*): Path for saving the output CSV/TSV file.
                        - **startBand** (*int*): Starting raster band to extract.
                        - **endBand** (*int*): Ending raster band to extract.

    :**Attributes**:    - **inputFile** (*str*): Input raster file path.
                        - **outputFile** (*str*): Output tabular file path.
                        - **startBand** (*int*): First band index to be extracted.
                        - **endBand** (*int*): Last band index to be extracted.
                        - **header** (*list*): List of column headers to be used in the output file: [x, y, band1, band2, ..., bandN]

    **Execution methods**

    .. code-block:: python

           from geoanalytics.conversion import Raster2CSV

           converter = Raster2CSV(inputFile='data.tif', outputFile='table.tsv', startBand=1, endBand=5)

           converter.run()

    **Credits**

    This implementation was created and revised under the guidance of Professor Rage Uday Kiran.
    """
    def __init__(self, inputFile, outputFile, startBand, endBand):
        """
        Constructor to initialize the Raster2CSV object with input and output parameters.

        :param inputFile: Raster file from which data needs to be extracted.
        :param outputFile: File path where the resulting CSV/TSV will be saved.
        :param startBand: The first raster band to read.
        :param endBand: The last raster band to read.
        """

        self.inputFile = inputFile
        self.outputFile = outputFile
        self.startBand = startBand
        self.endBand = endBand
        #self.header = ['coordinate'] + [f'-band{band}' for band in range(startBand, endBand + 1)]
        self.header = ['x', 'y'] + [f'{band}' for band in range(startBand, endBand + 1)]

    def run(self):
        """
        Executes the raster to CSV/TSV conversion process.

        1. Removes the output file if it already exists.
        2. Calls the `raster2tsv` utility to extract raster data for specified bands.
        3. Reads the TSV file into a pandas DataFrame.
        4. Sets the proper column headers.
        5. Writes the final result back to the output file.
        """

        if os.path.exists(self.outputFile):
            os.remove(self.outputFile)

        band_args = ' '.join([f'-band {b}' for b in range(self.startBand, self.endBand + 1)])
        params = f"{band_args} {self.inputFile} {self.outputFile}"

        print(f"Processing: {self.inputFile}")
        raster2tsv.raster2tsv(params)  # class call with CLI-style string

        df = pd.read_csv(self.outputFile, header=None, sep='\t')
        df.columns = self.header
        df.to_csv(self.outputFile, index=False, sep='\t')

        print(f"Done. Output saved to: {self.outputFile}")