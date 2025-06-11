# Img2Tiff is a utility that converts raster image formats (e.g., .img) into GeoTIFF (.tif) format using GDAL.
#
# **Importing and Using the Img2Tiff Class in a Python Program**
#
#         from geoanalytics.conversion import Img2Tiff
#
#         converter = Img2Tiff(inputFile='input.img', outputFile='output.tif')
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

class Img2Tiff:
    """
    **About this algorithm**

    :**Description**: Img2Tiff is a lightweight converter tool that transforms raster image files (commonly `.img` or other formats supported by GDAL) into GeoTIFF (`.tif`) format.

    :**Parameters**:    - **inputFile** (*str*): Path to the input raster image file (e.g., .img).
                        - **outputFile** (*str*): Path to the desired output GeoTIFF file (.tif).

    :**Attributes**:    - **inputFile** (*str*): The input file path provided by the user.
                        - **outputFile** (*str*): The desired output file path for the converted TIFF.
                        - **dataset** (*gdal.Dataset*): GDAL dataset object created during the file load process.

    **Execution methods**

    .. code-block:: python

            from geoanalytics.conversion import Img2Tiff

            converter = Img2Tiff("source.img", "converted.tif")

            converter.run()

    **Credits**

    This implementation was created    and revised      under the guidance of Professor Rage Uday Kiran.
    """

    def __init__(self, inputFile, outputFile):
        """
        Constructor to initialize the Img2Tiff converter.

        :param inputFile: Full path to the input .img or raster file.
        :param outputFile: Full path to save the resulting .tif file.
        """

        self.inputFile = inputFile
        self.outputFile = outputFile
        self.dataset = None

    def load(self):
        """
        Loads the input raster image using GDAL.

        :raises FileNotFoundError: If the input file cannot be opened.
        """

        self.dataset = gdal.Open(self.inputFile)
        if not self.dataset:
            raise FileNotFoundError(f"Could not open file: {self.inputFile}")

    def run(self):
        """
        Executes the conversion process from input raster to GeoTIFF using GDAL Translate.

        This method loads the dataset (if not already loaded) and performs the conversion.
        """

        if self.dataset is None:
            self.load()
        gdal.Translate(self.outputFile, self.dataset)
        print(f"Converted {self.inputFile} to {self.outputFile}")
