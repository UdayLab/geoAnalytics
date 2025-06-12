# raster2tsv is a raster data extraction and conversion utility that reads selected bands from a raster file (e.g., GeoTIFF),extracts geospatial coordinates and pixel values, and exports the information into a tab-separated values (TSV) format.
#
# **Importing and Using the raster2tsv Class in a Python Program**
#
#             from geoanalytics.conversion import raster2tsv
#
#             cmd = "-band 1 -band 2 -srcwin 0 0 100 100 -skip 2 raster_input.tif output.tsv"
#
#             raster2tsv(cmd)

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


try:
    from osgeo import gdal
except ImportError:
    import gdal

import sys

try:
    import numpy as Numeric
except ImportError:
    import Numeric


# =============================================================================
def Usage():
    print('Usage: raster2tsv.py [-skip factor] [-srcwin xoff yoff width height]')
    print('                   [-band b] srcfile [dstfile]')
    print('')
    sys.exit(1)


# =============================================================================
#
# Program mainline.
#
class raster2tsv:
    """
    **About this algorithm**

    :**Description**: raster2tsv is a raster data extraction and conversion utility that reads selected bands from a raster file (e.g., GeoTIFF),extracts geospatial coordinates and pixel values, and exports the information into a tab-separated values (TSV) format.

    :**Parameters**:    - **-band b (*int*, repeatable): Specifies which raster bands to extract. If not provided, defaults to band 1.
                        - **-srcwin xoff yoff width height** (*int x 4*, optional): Defines a rectangular sub-window (offset and size) of the raster to process.
                        - **-skip factor** (*int*, optional): Sets a sampling factor, allowing the program to skip pixels for faster processing.
                        - **srcfile** (*str*): Path to the input raster file (e.g., `.tif` or `.img`).
                        - **dstfile** (*str*, optional): Path to the output `.tsv` file. If not provided, output is printed to standard output.

    :**Attributes**:    - **srcfile** (*str*): Raster file being read.
                        -**dstfile** (*str*): Output TSV file receiving the data.
                        - **band_nums** (*list[int]*): List of band indices to extract.
                        - **srcwin** (*tuple[int, int, int, int]*): Tuple representing the rectangular window to process (x offset, y offset, width, height).
                        - **skip** (*int*): Sampling interval for skipping pixels.
                        - **gt** (*tuple*): GeoTransform tuple used to compute real-world coordinates (x, y).

    **Execution methods**

    **Calling from a Python program**

    .. code-block:: python

            from geoanalytics.conversion import raster2tsv

            cmd = "-band 1 -band 2 -srcwin 0 0 100 100 -skip 2 raster_input.tif output.tsv"

            raster2tsv(cmd)


     **Credits**

    This implementation was created and revised under the guidance of Professor Rage Uday Kiran.
    """

    def __init__(self, parameter):
        """
        Initializes the raster2tsv object and performs the raster-to-TSV conversion process.

        :param parameter: A string of space-separated command-line-style arguments.
        Expected format includes:
            - '-band b' (int): Specify raster band(s) to extract.
            - '-srcwin xoff yoff width height' (int x 4): Define subwindow to read.
            - '-skip factor' (int): Skip factor for sampling (e.g., 2 means process every second pixel).
            - 'srcfile' (str): Path to the raster file.
            - 'dstfile' (str, optional): Path to save the TSV output.

        The method reads the specified bands from the raster file, computes the geospatial coordinates, extracts pixel values, and writes them in a tab-separated format to the given output file (or stdout).
        """
        # self.parameter = parameter
        srcwin = None
        skip = 1
        srcfile = None
        dstfile = None
        band_nums = []

        gdal.AllRegister()
        # argv = gdal.GeneralCmdLineProcessor(parameter)
        argv = parameter.split(" ")
        #print(argv)
        if argv is None:
            sys.exit(0)

        # Parse command line arguments.
        i = 0
        while i < len(argv):
            arg = argv[i]
            #print(arg)
            if arg == '-srcwin':
                srcwin = (int(argv[i + 1]), int(argv[i + 2]),
                          int(argv[i + 3]), int(argv[i + 4]))
                i = i + 4

            elif arg == '-skip':
                skip = int(argv[i + 1])
                i = i + 1

            elif arg == '-band':
                band_nums.append(int(argv[i + 1]))
                i = i + 1



            elif arg[0] == '-':
                Usage()



            elif srcfile is None:
                srcfile = arg


            elif dstfile is None:
                dstfile = arg

            else:
                Usage()

            i = i + 1

        if srcfile is None:
            Usage()
        if band_nums == []: band_nums = [1]
        # Open source file.
        srcds = gdal.Open(srcfile)
        if srcds is None:
            print('Could not open %s.' % srcfile)
            sys.exit(1)

        bands = []
        for band_num in band_nums:
            band = srcds.GetRasterBand(band_num)

            if band is None:
                print('Could not get band %d' % band_num)
                sys.exit(1)
            bands.append(band)

        gt = srcds.GetGeoTransform()

        # Collect information on all the source files.
        if srcwin is None:
            srcwin = (0, 0, srcds.RasterXSize, srcds.RasterYSize)

        # Open the output file.
        if dstfile is not None:
            dst_fh = open(dstfile, 'a')
        else:
            dst_fh = sys.stdout
        band_format = ("%g " * len(bands)).rstrip()

        # Setup an appropriate print format.
        if abs(gt[0]) < 180 and abs(gt[3]) < 180 \
                and abs(srcds.RasterXSize * gt[1]) < 180 \
                and abs(srcds.RasterYSize * gt[5]) < 180:
            format = '%.10g %.10g %s'

        else:
            format = '%.3f %.3f %s'

        # Loop emitting data.

        for y in range(srcwin[1], srcwin[1] + srcwin[3], skip):

            data = []
            for band in bands:
                band_data = band.ReadAsArray(srcwin[0], y, srcwin[2], 1)
                band_data = Numeric.reshape(band_data, (srcwin[2],))
                data.append(band_data)

            for x_i in range(0, srcwin[2], skip):

                x = x_i + srcwin[0]

                geo_x = gt[0] + (x + 0.5) * gt[1] + (y + 0.5) * gt[2]
                geo_y = gt[3] + (x + 0.5) * gt[4] + (y + 0.5) * gt[5]

                x_i_data = []
                for i in range(len(bands)):
                    x_i_data.append(data[i][x_i])

                band_str = band_format % tuple(x_i_data)

                line1 = format % (float(geo_x), float(geo_y), band_str)
                li = list(line1.split(" "))

                a = str(li[2:]).replace('\'', '').replace('[', '').replace(']', '')
                #print(a)
                temp_line = f"{li[0]}\t{li[1]}\t"#'Point(' +li[0] + ' ' + li[1] + ')\t'
                for a in li[2:]:
                    temp_line += str(a) + '\t'#temp_line = temp_line + str(a) + '\t'
                dst_fh.write(temp_line.rstrip('\t') + "\n")#dst_fh.write(temp_line[:-1] + "\n")