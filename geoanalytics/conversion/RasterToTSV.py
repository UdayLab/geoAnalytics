import os
import pandas as pd
from geoanalytics.conversion import raster2tsv

class RasterToTSV:
    def __init__(self, input_file, output_file, start_band, end_band):
        self.input_file = input_file
        self.output_file = output_file
        self.start_band = start_band
        self.end_band = end_band
        #self.header = ['coordinate'] + [f'-band{band}' for band in range(start_band, end_band + 1)]
        self.header = ['x', 'y'] + [f'-band{band}' for band in range(start_band, end_band + 1)]

    def convert(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        band_args = ' '.join([f'-band {b}' for b in range(self.start_band, self.end_band + 1)])
        params = f"{band_args} {self.input_file} {self.output_file}"

        print(f"Processing: {self.input_file}")
        raster2tsv.raster2tsv(params)  # class call with CLI-style string

        df = pd.read_csv(self.output_file, header=None, sep='\t')
        df.columns = self.header
        df.to_csv(self.output_file, index=False, sep='\t')

        print(f"Done. Output saved to: {self.output_file}")
