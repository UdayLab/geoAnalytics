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

import matplotlib.pyplot as plt
import rasterio


class NetCDFViewer():
    def __init__(self, inputFile):
        self.inputFile = inputFile
        self.imageData = None

    def run(self, cmap='gray', title='NC Image'):
        with rasterio.open(self.inputFile) as src:
            self.imageData = src.read(1)
        if self.imageData is None:
            raise ValueError("Image not loaded")
        plt.imshow(self.imageData, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.show()