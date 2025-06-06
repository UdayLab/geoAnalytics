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

    def __init__(self, parameter):
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