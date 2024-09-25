import subprocess
import sys
from osgeo import gdal
import pandas as pd


class RasterConverter:
    """
    :Description: Convert raster data to CSV format and apply scale factor and offset to the converted data.

    Attributes:
        input_dataset (str): Path to the input raster dataset.
        output_csv (str): Path to the output CSV file.
        dataset (raster Dataset): GDAL dataset object representing the input raster dataset.
        raster_df (dataframe): DataFrame to store the converted data.


    """

    def __init__(self, input_dataset, output_csv):
        """
        Initializes the RasterConverter.

        Args:
            input_dataset (str): Path to the input raster dataset.
            output_csv (str): Path to the output CSV file.
        """
        self.input_dataset = input_dataset
        self.output_csv = output_csv
        self.dataset = None
        self.raster_df = None

    def getScalefactorOffset(self, band_number):
        """
        Retrieves the scale factor and offset for a given band.

        Args:
            band_number (int): The band number.

        Returns:
            tuple: The scale factor and offset for the band.
        """
        band = self.dataset.GetRasterBand(int(band_number))
        scale_factor = band.GetScale()
        offset = band.GetOffset()
        return scale_factor, offset

    def applyScaleOffset(self):
        """
        Applies the scale factor and offset to the converted data.
        """
        for i in range(len(self.raster_df.columns[2:])):
            scale_factor, offset = self.getScalefactorOffset(i + 1)
            self.raster_df[i + 2] = self.raster_df[i + 2] * scale_factor + offset

    def convert2csv(self):
        """
        Converts the raster dataset to CSV format.
        """
        command = ['gdal2xyz.py', '-all', self.input_dataset, self.output_csv]
        subprocess.call(command)

    def formatCoordinates(self):
        """
        Formats the first two columns of the DataFrame to "Point(x, y)" format.
        """
        self.raster_df[0] = "Point(" + self.raster_df[0].astype(str) + ","
        self.raster_df[1] = self.raster_df[1].astype(str) + ")"
        # self.raster_df[0] = self.raster_df[0].str.replace(r'\s', '')
        # self.raster_df[1] = self.raster_df[1].str.replace(r'\s', '')

    def runConversion(self):
        """
        Runs the entire conversion process.
        """
        self.convert2csv()

        self.raster_df = pd.read_csv(self.output_csv, header=None, sep=' ')

        self.dataset = gdal.Open(self.input_dataset)

        self.applyScaleOffset()

        self.formatCoordinates()

        print(self.raster_df.head())

        return self.raster_df


if __name__ == "__main__":
    input_dataset = sys.argv[1]
    output_csv = sys.argv[2]
    converter = RasterConverter(input_dataset, output_csv)
    rasterDf = converter.runConversion()
    rasterDf.to_csv(output_csv, sep=' ', index=None, header=None)