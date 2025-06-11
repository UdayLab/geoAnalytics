# LBL2Tiff is a converter that transforms .LBL (labelled raster) files into GeoTIFF (.tif) format using the GDAL library.
#
# **Importing and Using the LBL2Tiff Class in a Python Program**
#
#         from geoanalytics.conversion import LBL2Tiff
#
#         converter = LBL2Tiff(inputFile='input.lbl', outputFile='output.tif')
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

class LBL2Tiff:
    """
    **About this algorithm**

    :**Description**: LBL2Tiff converts raster images with `.LBL` format (labelled data, typically from planetary datasets) to the standardized GeoTIFF (`.tif`) format using the GDAL library.

    :**Parameters**:    - **inputFile** (*str*): Path to the input LBL or compatible raster file.
                        - **outputFile** (*str*): Path where the converted GeoTIFF file will be saved.

    :**Attributes**:    - **inputFile** (*str*): The path to the original input file provided by the user.
                        - **outputFile** (*str*): The desired output path for the resulting TIFF file.
                        - **dataset** (*gdal.Dataset*): The loaded raster dataset object initialized by GDAL.

    **Execution methods**

    .. code-block:: python

            from geoanalytics.conversion import LBL2Tiff

            converter = LBL2Tiff("image.lbl", "converted_image.tif")

            converter.run()

    **Credits**

    This implementation was created     and revised      under the guidance of Professor Rage Uday Kiran.
    """

    def __init__(self, inputFile, outputFile):
        """
        Constructor to initialize the LBL2Tiff conversion object.

        :param inputFile: Path to the input .LBL or raster file.
        :param outputFile: Path where the output .tif file will be saved.
        """

        self.inputFile = inputFile
        self.outputFile = outputFile
        self.dataset = None

    def load(self):
        """
        Loads the input raster dataset using GDAL.

        :raises FileNotFoundError: If GDAL fails to open the input file.
        """

        self.dataset = gdal.Open(self.inputFile)
        if not self.dataset:
            raise FileNotFoundError(f"Could not open file: {self.inputFile}")

    def run(self):
        """
        Performs the file format conversion from LBL to TIFF using GDAL Translate.

        Loads the dataset if not already loaded and converts it to the output file.
        """
        if self.dataset is None:
            self.load()
        gdal.Translate(self.outputFile, self.dataset)
        print(f"Converted {self.inputFile} to {self.outputFile}")
