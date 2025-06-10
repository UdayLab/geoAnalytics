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
    def __init__(self, inputFile, outputFile, startBand, endBand):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.startBand = startBand
        self.endBand = endBand
        #self.header = ['coordinate'] + [f'-band{band}' for band in range(startBand, endBand + 1)]
        self.header = ['x', 'y'] + [f'{band}' for band in range(startBand, endBand + 1)]

    def run(self):
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