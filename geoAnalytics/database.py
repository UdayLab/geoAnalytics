from geoAnalytics import csv2raster
from geoAnalytics.config import config
from shapely import geos, wkb, wkt
from osgeo import gdal
import numpy as Numeric
import pandas as pd
import subprocess
import psycopg2
import random
import glob
import sys
import os
import threading
from tqdm import tqdm


def connect(dbName, hostIP, user, password, port=5432):
    """
    Connect to the database using the parameters.

    :param dbName: name of the database
    :param hostIP: host IP
    :param user: user name
    :param password: password
    :param port: port
    """

    if dbName != "":
        with open('database.ini', "w") as dbFile:
            dbFile.write("[postgresql]\nhost = " + str(hostIP) + "\nport = " + str(port) +
                         "\ndatabase = " + str(dbName) + "\nuser = " + str(user) + "\npassword = " + str(password))
            dbFile.close()
    connection = testConnection()


def disconnect():
    """
    Disconnect from the database. 

    """
    conn = None
    # read database configuration
    params = config()
    # connect to the PostgreSQL database
    conn = psycopg2.connect(**params)
    # create a new cursor
    curr = conn.cursor()
    if conn is not None:
        print("Disconnecting from repository")
        conn.close()
        print("Disconnected from repository")

# def reConnect(self, dbName="", hostIP="", user="", password="", port=5432):
#     """
#     Edit the connection to the database
#
#     :param dbName: name of the database
#     :param hostIP: host IP
#     :param user: user name
#     :param password: password
#     :param port: port
#     """
#     conn = None
#     if dbName != "":
#         with open('ini', "w") as dbFile:
#             buffer = "[postgresql]\nhost = " + hostIP + "\nport = " + str(
#                 port) + "\ndatabase = " + dbName + "\nuser = " + user + "\npassword = " + password
#             dbFile.write(buffer)
#             dbFile.close()
#     self.testConnection()


def testConnection():
    """
    Test the connection to the database
    """

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        curr = conn.cursor()
        curr.execute("select version();")
        for item in curr:
            print(item)
        print('You are now connected')
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def createRepository(repositoryName, totalBands, SRID=4326):
    """
    Create a repository in the database

    :param repositoryName: name of the repository
    :param totalBands: total number of bands
    :param SRID: spatial reference ID
    """

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        curr = conn.cursor()
        # total bands is number
        query = "geog geometry(POINT," + str(SRID) + "),"
        for i in range(1, totalBands):
            query += 'b' + str(i) + ' float,'
        query += 'b' + str(totalBands) + ' float'
        curr.execute("create table " + repositoryName + "(" + query + ")")
        curr.execute("CREATE INDEX index_" + repositoryName +
                     " ON " + repositoryName + " USING GIST (geog)")
        conn.commit()
        print('Repository created')
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')


def deleteRepository(repositoryName):
    """
    Delete a repository from the database

    :param repositoryName: name of the repository
    """

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        curr = conn.cursor()
        curr.execute("drop table " + repositoryName + ";")
        conn.commit()
        print(repositoryName + ' deleted successfully')
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')


def cloneRepository(repositoryName, cloneRepositoryName):
    """
    Clone a repository from the database

    :param repositoryName: name of the repository
    :param cloneRepositoryName: name of the cloned repository
    """
    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        curr = conn.cursor()
        curr.execute("create table " + cloneRepositoryName +
                     " as (select * from " + repositoryName + ");")
        conn.commit()
        print('Repository cloned')
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')


def renameRepository(repositoryName, newRepositoryName):
    """
    Change the name of a repository in the database

    :param repositoryName: name of the repository
    :param newRepositoryName: new name of the repository
    """
    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        curr = conn.cursor()
        curr.execute("ALTER TABLE IF EXISTS " +
                     repositoryName + " RENAME TO " + newRepositoryName + ";")
        conn.commit()
        print(repositoryName + ' changed to ' + newRepositoryName)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')


def insertRaster(repositoryName, fileName, totalBands, scalingFactor, SRID=4326):
    """
    Insert a TIFF file into the database

    :param repositoryName: name of the Repository
    :param fileName: name of the TIFF file
    :param totalBands: number of bands
    :param SRID: spatial reference ID
    :param scalingFactor: scaling factor
    """
    tempFile = _r2tsv(totalBands, fileName, scalingFactor, SRID)
    insertCSV(tempFile, repositoryName)
    if os.path.exists(tempFile):
        os.remove(tempFile)


def insertRasterFolder(repositoryName, folderName, totalBands, scalingFactor, extension=".lbl",SRID=4326):
    """
    Insert a TIFF file into the database

    :param repositoryName: name of the Repository
    :param fileName: name of the TIFF file
    :param totalBands: number of bands
    :param SRID: spatial reference ID
    :param scalingFactor: scaling factor
    """
    files = []
    for file in os.listdir(folderName):
        if file.endswith(extension):
            files.append(file)

#     print(files)

    # single thread
    for i in tqdm(range(len(files))):
        tempFile = _r2tsv(totalBands, folderName + '/' + files[i], scalingFactor, SRID)
        insertCSV(tempFile, repositoryName)
        if os.path.exists(tempFile):
            os.remove(tempFile)

    # multiprocessing

    # threads = []
    # for file in files:
    #     t = threading.Thread(target=insertRaster, args=(
    #         repositoryName, folderName + '/' + file, totalBands, scalingFactor, SRID))
    #     threads.append(t)
    #     t.start()

    # for t in threads:
    #     t.join()


def insertCSV(filename, repositoryName, seperator=' '):
    """
    Insert a CSV file into the database

    :param filename: name of the CSV file
    :param repositoryName: name of the Repository
    :param seperator: seperator
    """

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        curr = conn.cursor()
        buffer = "COPY " + str(repositoryName) + " FROM '" + str(os.getcwd()) + '/' + str(
            filename) + "' DELIMITER '" + str(seperator) + "' CSV;"
        curr.execute(buffer)
        conn.commit()
#         print('File inserted')
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
#             print('Repository connection closed.')


def _r2tsv(endBand, srcfile, scalingFactor, SRID):
    """
    Convert a raster to a tsv file

    :param startBand: start band
    :param endBand: end band
    :param srcfile: source file
    :param dstfile: destination file
    :param scalingFactor: scaling factor
    """

    tempFile = str(random.randint(0, 100000)) + ".txt"
    band_nums = range(1, endBand + 1)
    srcwin = None
    if band_nums == []:
        band_nums = [1]

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
    if tempFile is not None:
        dst_fh = open(tempFile, 'wt')
    else:
        dst_fh = sys.stdout
    band_format = ("%g " * len(bands)).rstrip() + '\n'

    # Setup an appropriate print format.
    if abs(gt[0]) < 180 and abs(gt[3]) < 180 \
            and abs(srcds.RasterXSize * gt[1]) < 180 \
            and abs(srcds.RasterYSize * gt[5]) < 180:
        format = '%s %s'
    else:
        format = '%s %s'

    # Loop emitting data.
    for y in range(srcwin[1], srcwin[1] + srcwin[3]):
        data = []
        for band in bands:
            band_data = band.ReadAsArray(srcwin[0], y, srcwin[2], 1)
            band_data = Numeric.reshape(band_data, (srcwin[2],))
            data.append(band_data)

        for x_i in range(0, srcwin[2]):
            x = x_i + srcwin[0]
            geo_x = gt[0] + (x + 0.5) * gt[1] + (y + 0.5) * gt[2]
            geo_y = gt[3] + (x + 0.5) * gt[4] + (y + 0.5) * gt[5]
            x_i_data = []
            for i in range(len(bands)):
                x_i_data.append(data[i][x_i]*scalingFactor)
            band_str = band_format % tuple(x_i_data)
            # line = format % (wkb.dumps(wkt.loads("POINT(" + str(float(geo_x)) + " " + str(float(geo_y)) + ")"),
            #                  hex=True, srid=SRID), band_str)  # Convert X and Y in hex, store the data, and upload
            line = format % (wkb.dumps(wkt.loads("POINT(" + str(round(geo_x, 4)) + " " + str(round(geo_y, 4)) + ")"),
                             hex=True, srid=SRID), band_str)  # Convert X and Y in hex, store the data, and upload
            dst_fh.write(line)
    return tempFile


# def addBandToRepository(self, repositoryName, bandFormula):
#     # Add a column to a table using alter Command
#     try:
#         conn = None
#         # read database configuration
#         params = config()
#         # connect to the PostgreSQL database
#         conn = psycopg2.connect(**params)
#         # create a new cursor
#         curr = conn.cursor()
#         # describe table command

#         curr.execute("ALTER TABLE " + repositoryName + " ADD COLUMN B" +
#                      str(bandNumber) + " float" + ";")
#         conn.commit()
#         print('Band added to repository')
#     except (Exception, psycopg2.DatabaseError) as error:
#         print(error)
#     finally:
#         if conn is not None:
#             conn.close()
#             print('Repository connection closed.')

def deleteBandInRepository(repositoryName, bandNumber):
    # This function will delete the band number attribute from the table.
    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        curr = conn.cursor()
        curr.execute("ALTER TABLE " + repositoryName + " DROP COLUMN B" +
                     str(bandNumber) + ";")
        conn.commit()
        print('Band deleted from repository')
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')

    return


def getRaster(repositoryName, rasterFileName, Xmin, Ymin, Xmax, Ymax, Bands="*"):
    """
    Get a raster image from the database

    :param repositoryName: name of the repository
    :param rasterFileName: name of the raster file
    :param Xmin: minimum X coordinate
    :param Ymin: minimum Y coordinate
    :param Xmax: maximum X coordinate
    :param Ymax: maximum Y coordinate
    :param Bands: bands to be extracted
    """
    # connect to database
    # create geoTIFF file
    # gDal transform to geoTIFF
    print('Getting raster from database')
    dataFrame2Raster(getDataframeForEnvelope(
        repositoryName, Xmin, Ymin, Xmax, Ymax, Bands), rasterFileName)


def dataFrame2Raster(dataframe, rasterFileName):
    """
    Create a raster image from a dataframe

    :param dataframe: dataframe
    :param rasterFileName: name of the raster file with no file extension.

    """
    print('Creating raster file')

    df = dataframe
    csv2raster.csv2raster(output_file=rasterFileName, dataframe=df)


def getDataframe(repositoryName, Bands="*", SRID=4326):
    """
    Get a dataframe from the database for a given envelope

    :param repositoryName: name of the repository
    :param Xmin: minimum X coordinate
    :param Ymin: minimum Y coordinate
    :param Xmax: maximum X coordinate
    :param Ymax: maximum Y coordinate
    :param Bands: bands to be extracted
    :param SRID: spatial reference ID
    """

    # connect to database
    # create geoTIFF file
    # gDal transform to geoTIFF
    print('Getting dataframe from database')

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        query = "SELECT ST_X(geog) as x, ST_Y(geog) as y, " + \
            Bands + " FROM " + str(repositoryName) + ";"
        dataFrameEnvelope = pd.read_sql_query(query, conn)
        print('Dataframe created')
        if Bands == "*":
            dataFrameEnvelope = dataFrameEnvelope.drop(columns=["geog"])
        return dataFrameEnvelope
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')


def getDataframe(repositoryName, Bands="*", SRID=4326):
    """
    Get a dataframe from the database for a given envelope

    :param repositoryName: name of the repository
    :param Bands: bands to be extracted
    :param SRID: spatial reference ID
    """

    # connect to database
    # create geoTIFF file
    # gDal transform to geoTIFF
    print('Getting dataframe from database')

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        query = "SELECT ST_X(geog) as x, ST_Y(geog) as y, " + \
            Bands + " FROM " + str(repositoryName) + ";"
        dataFrameEnvelope = pd.read_sql_query(query, conn)
        print('Dataframe created')
        if Bands == "*":
            dataFrameEnvelope = dataFrameEnvelope.drop(columns=["geog"])
        return dataFrameEnvelope
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')


def getDataframeForEnvelope(repositoryName, Xmin, Ymin, Xmax, Ymax, Bands="*", SRID=4326):
    """
    Get a dataframe from the database for a given envelope

    :param repositoryName: name of the repository
    :param Xmin: minimum X coordinate
    :param Ymin: minimum Y coordinate
    :param Xmax: maximum X coordinate
    :param Ymax: maximum Y coordinate
    :param Bands: bands to be extracted
    :param SRID: spatial reference ID
    """

    # connect to database
    # create geoTIFF file
    # gDal transform to geoTIFF
    print('Getting dataframe from database')

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        query = "SELECT ST_X(geog) as x, ST_Y(geog) as y, " + Bands + " FROM " + str(repositoryName) + " WHERE " + str(
            repositoryName) + ".geog && ST_MakeEnvelope(" + str(Xmin) + ',' + str(Ymin) + ',' + str(Xmax) + ',' + str(Ymax) + ");"
        dataFrameEnvelope = pd.read_sql_query(query, conn)
        print('Dataframe created')
        if Bands == "*":
            dataFrameEnvelope = dataFrameEnvelope.drop(columns=["geog"])
        return dataFrameEnvelope
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')


def kNearestPixels(repositoryName, X, Y, k=1000, Bands="*", SRID=4326):
    """
    Get a dataframe from the database for a point and its neighbors

    :param repositoryName: name of the repository
    :param X: X coordinate
    :param Y: Y coordinate
    :param k: number of neighbours
    :param Bands: bands to be extracted
    :param SRID: spatial reference ID
    """
    print('Getting dataframe from database')

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        query = "SELECT ST_X(geog) as x, ST_Y(geog) as y, " + str(Bands) + " FROM " + str(
            repositoryName) + " ORDER BY " + str(repositoryName) + ".geog <-> 'SRID=" + str(SRID) + ";POINT(" + str(
            X) + ' ' + str(Y) + ")' limit " + str(k) + ";"
        dataFrameEnvelope = pd.read_sql_query(query, conn)
        print('Dataframe created')
        if Bands == "*":
            dataFrameEnvelope = dataFrameEnvelope.drop(columns=["geog"])
        return dataFrameEnvelope
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')


def getRasterForkNearestPixels(repositoryName, rasterFileName="rasterFile.nc", X=0, Y=0, k=1000, Bands="*"):
    """
    Get a raster image from the database for a point and its neighbors

    :param repositoryName: name of the repository
    :param rasterFileName: name of the raster file
    :param X: X coordinate
    :param Y: Y coordinate
    :param k: number of neighbours
    :param Bands: bands to be extracted
    """
    # connect to database
    # create geoTIFF file
    # gDal transform to geoTIFF
    df = kNearestPixels(repositoryName, X, Y, k, Bands)
    dataFrame2Raster(df, rasterFileName)


def getRepositorySize(tableName):
    """

    """

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        query = "SELECT pg_size_pretty( pg_total_relation_size('" + \
            tableName + "') );"
        dataFrameEnvelope = pd.read_sql_query(query, conn)
        print(dataFrameEnvelope.iloc[0]['pg_size_pretty'])
        return dataFrameEnvelope.iloc[0]['pg_size_pretty']
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')


def getSizeOfAllRepositories():
    """
    Get return the size of the repository/database/table

    :param repositoryName: name of the repository
    """

    try:
        conn = None
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        q1 = "select current_database();"
        q2 = pd.read_sql_query(q1, conn)

        query = " SELECT pg_size_pretty( pg_database_size('" + \
            q2.iloc[0]['current_database'] + "') );"
        dataFrameEnvelope = pd.read_sql_query(query, conn)
        print(dataFrameEnvelope.iloc[0]['pg_size_pretty'])
        return dataFrameEnvelope.iloc[0]['pg_size_pretty']
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Repository connection closed.')
