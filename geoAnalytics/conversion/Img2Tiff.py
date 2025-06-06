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

class Img2Tiff:
    def __init__(self, inputFile, outputFile):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.dataset = None

    def load(self):
        self.dataset = gdal.Open(self.inputFile)
        if not self.dataset:
            raise FileNotFoundError(f"Could not open file: {self.inputFile}")

    def convert(self):
        if self.dataset is None:
            self.load()
        gdal.Translate(self.outputFile, self.dataset)
        print(f"Converted {self.inputFile} to {self.outputFile}")
