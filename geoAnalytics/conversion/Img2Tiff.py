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
