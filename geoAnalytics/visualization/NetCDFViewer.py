import matplotlib.pyplot as plt
import rasterio


class NetCDFViewer():
    def __init__(self, tiffFile):
        self.tiffFile = tiffFile
        self.imageData = None

    def displayImage(self, cmap='gray', title='TIFF Image'):
        with rasterio.open(self.tiffFile) as src:
            self.imageData = src.read(1)
        if self.imageData is None:
            raise ValueError("Image not loaded")
        plt.imshow(self.imageData, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.show()