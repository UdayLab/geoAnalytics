from geoanalytics import csv2raster
from geoanalytics.config import config
from shapely import wkb, wkt
from osgeo import gdal
import numpy as np
import pandas as pd
import psycopg2
import random
import os
import sys
from tqdm import tqdm
import multiprocessing
from scoreCalculator import ScoreCalculator

class GeoDatabaseManager:
    def __init__(self):
        self.conn = None
        self.scores = None

    def connect(self, dbName = None, hostIP = None, user = None, password = None, port=5432):
        """
        Connect to the database using the parameters.
        """
        if dbName != None:
            with open('database.ini', "w") as dbFile:
                dbFile.write("[postgresql]\nhost = " + str(hostIP) + "\nport = " + str(port) +
                             "\ndatabase = " + str(dbName) + "\nuser = " + str(user) + "\npassword = " + str(password))
        self.conn = self.test_connection()

    def disconnect(self):
        """
        Disconnect from the database.
        """
        if self.conn:
            print("Disconnecting from repository")
            self.conn.close()
            print("Disconnected from repository")

    def test_connection(self):
        """
        Test the connection to the database.
        """
        try:
            params = config()
            conn = psycopg2.connect(**params)
            curr = conn.cursor()
            curr.execute("select version();")
            for item in curr:
                print(item)
            print('You are now connected')
            return conn
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def create_repository(self, repositoryName, totalBands, SRID=4326):
        """
        Create a repository in the database.
        """
        try:
            params = config()
            self.conn = psycopg2.connect(**params)
            curr = self.conn.cursor()
            query = "geog geometry(POINT," + str(SRID) + "),"
            for i in range(1, totalBands):
                query += 'b' + str(i) + ' float,'
            query += 'b' + str(totalBands) + ' float'
            curr.execute(f"create table {repositoryName}({query})")
            curr.execute(f"CREATE INDEX index_{repositoryName} ON {repositoryName} USING GIST (geog)")
            self.conn.commit()
            print('Repository created')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            self.disconnect()

    def delete_repository(self, repositoryName):
        """
        Delete a repository from the database.
        """
        try:
            params = config()
            self.conn = psycopg2.connect(**params)
            curr = self.conn.cursor()
            curr.execute(f"drop table {repositoryName};")
            self.conn.commit()
            print(f"{repositoryName} deleted successfully")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            self.disconnect()
    
    def list_repositories(self):
        """
        List all repositories (tables) in the database.
        """
        try:
            self.connect()
            curr = self.conn.cursor()
            curr.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """
            )
            repositories = curr.fetchall()
            print("Available Repos:\n---------------")
            for repository in repositories:
                print(repository[0])
            print("---------------")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            self.disconnect()

    def clone_repository(self, repositoryName, cloneRepositoryName):
        """
        Clone a repository from the database.
        """
        try:
            params = config()
            self.conn = psycopg2.connect(**params)
            curr = self.conn.cursor()
            curr.execute(f"create table {cloneRepositoryName} as (select * from {repositoryName});")
            self.conn.commit()
            print('Repository cloned')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            self.disconnect()

    def rename_repository(self, repositoryName, newRepositoryName):
        """
        Change the name of a repository in the database.
        """
        try:
            params = config()
            self.conn = psycopg2.connect(**params)
            curr = self.conn.cursor()
            curr.execute(f"ALTER TABLE IF EXISTS {repositoryName} RENAME TO {newRepositoryName};")
            self.conn.commit()
            print(f"{repositoryName} changed to {newRepositoryName}")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            self.disconnect()
            
    def read_raster(self, fileName, totalBands, scalingFactor, SRID=4326):
        file = f"/tmp/{random.randint(0, 100000)}.txt"
        tempFile = self._r2tsv(totalBands, fileName, scalingFactor, SRID, file)
        df = pd.read_csv(tempFile, header = None, delimiter = " ")
        df.columns = ["WKB Position"] + ["B" + str(x) for x in range(totalBands)]
        return df
    
    def convertWKB(self, df):
        def wkb_to_geom(wkb_hex):
            return wkb.loads(bytes.fromhex(wkb_hex))

        # Apply the function to the 'wkb' column
        df['geometry'] = df['WKB Position'].apply(wkb_to_geom)

        # Extract longitude and latitude
        df['longitude'] = df['geometry'].apply(lambda geom: geom.x)
        df['latitude'] = df['geometry'].apply(lambda geom: geom.y)
        return df

    def insert_raster(self, repositoryName, fileName, totalBands, scalingFactor, SRID=4326):
        """
        Insert a TIFF file into the database.
        """
        tempFile = self._r2tsv(totalBands, fileName, scalingFactor, SRID)
        self.insert_csv(tempFile, repositoryName)

    def insert_raster_folder(self, repositoryName, folderName, totalBands, scalingFactor, extension=".lbl", SRID=4326, threads=16):
        """
        Insert TIFF files from a folder into the database.
        """
        files = [file for file in os.listdir(folderName) if file.endswith(extension)]
        print(files)
        
        pbar = tqdm(total=len(files))
        pool = multiprocessing.Pool(threads)
        
        def update(*a):
            pbar.update()
        
        for file in files:
            pool.apply_async(self.insert_raster, args=(repositoryName, os.path.join(folderName, file), totalBands, scalingFactor, SRID), callback=update)
        
        pool.close()
        pool.join()

    def insert_csv(self, filename, repositoryName, separator=' '):
        """
        Insert a CSV file into the database.
        """
        try:
            params = config()
            self.conn = psycopg2.connect(**params)
            curr = self.conn.cursor()
            # buffer = f"COPY {repositoryName} FROM '{filename}' DELIMITER '{separator}' CSV;"
            # buffer = f"\copy {repositoryName} FROM '{os.getcwd()}/{filename}' DELIMITER '{separator}' CSV;"
            with open(filename, 'r') as f:
                curr.copy_expert(f"COPY {repositoryName} FROM STDIN DELIMITER '{separator}' CSV", f)

            # curr.execute(buffer)
            self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            self.disconnect()

    def _r2tsv(self, endBand, srcfile, scalingFactor, SRID, tempFile = ""):
        """
        Convert a raster to a TSV file.
        """
        if tempFile == "":
            tempFile = f"/tmp/{random.randint(0, 100000)}.txt"
        band_nums = range(1, endBand + 1)

        srcds = gdal.Open(srcfile)
        if srcds is None:
            print(f'Could not open {srcfile}.')
            sys.exit(1)

        bands = [srcds.GetRasterBand(band_num) for band_num in band_nums]
        gt = srcds.GetGeoTransform()
        srcwin = (0, 0, srcds.RasterXSize, srcds.RasterYSize)

        with open(tempFile, 'wt') as dst_fh:
            band_format = ("%g " * len(bands)).rstrip() + '\n'

            for y in range(srcwin[1], srcwin[1] + srcwin[3]):
                data = [band.ReadAsArray(srcwin[0], y, srcwin[2], 1).reshape((srcwin[2],)) for band in bands]

                for x_i in range(0, srcwin[2]):
                    x = x_i + srcwin[0]
                    geo_x = gt[0] + (x + 0.5) * gt[1] + (y + 0.5) * gt[2]
                    geo_y = gt[3] + (x + 0.5) * gt[4] + (y + 0.5) * gt[5]
                    x_i_data = [data[i][x_i] * scalingFactor for i in range(len(bands))]
                    band_str = band_format % tuple(x_i_data)
                    line = f'{wkb.dumps(wkt.loads(f"POINT({round(geo_x, 4)} {round(geo_y, 4)})"), hex=True, srid=SRID)} {band_str}'
                    dst_fh.write(line)

        return tempFile

    def add_band(self, repositoryName, bandFormula):
        """
        Add a column to a table using the ALTER command.
        """
        try:
            params = config()
            self.conn = psycopg2.connect(**params)
            curr = self.conn.cursor()
            curr.execute(f"ALTER TABLE {repositoryName} ADD COLUMN B{bandFormula} float;")
            self.conn.commit()
            print('Band added to repository')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            self.disconnect()

    def delete_band(self, repositoryName, bandNumber):
        """
        Delete the band number attribute from the table.
        """
        try:
            params = config()
            self.conn = psycopg2.connect(**params)
            curr = self.conn.cursor()
            curr.execute(f"ALTER TABLE {repositoryName} DROP COLUMN B{bandNumber};")
            self.conn.commit()
            print('Band deleted from repository')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            self.disconnect()

    def get_raster(self, repositoryName, rasterFileName, Xmin, Ymin, Xmax, Ymax, Bands="*"):
        """
        Get a raster image from the database.
        """
        print('Getting raster from database')
        df = self.get_dataframe_for_envelope(repositoryName, Xmin, Ymin, Xmax, Ymax, Bands)
        self.dataframe_to_raster(df, rasterFileName)

    def dataframe_to_raster(self, dataframe, rasterFileName):
        """
        Create a raster image from a dataframe.
        """
        print('Creating raster file')
        csv2raster.csv2raster(output_file=rasterFileName, dataframe=dataframe)

    def get_dataframe_for_envelope(self, repositoryName, Xmin, Ymin, Xmax, Ymax, Bands="*"):
        """
        Get a dataframe for an envelope.
        """
        # ST_X(geog) as x, ST_Y(geog) as y
        sql = "SELECT ST_X(geog) as x, ST_Y(geog) as y, " + Bands + " FROM " + str(repositoryName) + " WHERE " + str(
            repositoryName) + ".geog && ST_MakeEnvelope(" + str(Xmin) + ',' + str(Ymin) + ',' + str(Xmax) + ',' + str(Ymax) + ");"
        params = config()
        self.conn = psycopg2.connect(**params)
        df = pd.read_sql(sql, self.conn)
        if Bands == "*":
            df.drop(columns=['geog'], inplace=True)
        return df
    
    def get_dataframe(self, repositoryName, Bands = "*"):
        """
        Get a dataframe for the entire repository.
        """
        sql = "SELECT ST_X(geog) as x, ST_Y(geog) as y, " + Bands + " FROM " + str(repositoryName) + ";"
        params = config()
        self.conn = psycopg2.connect(**params)
        
        df = pd.read_sql(sql, self.conn)
        if Bands == "*":
            df.drop(columns=['geog'], inplace=True)
        return df
    
    def filter(self, filterFile):
        """
        Filter the dataframe.
        """
        if self.scores is None:
            df = pd.read_csv(filterFile, header=None, delimiter=" ", dtype=float)
            self.scores = [ScoreCalculator(df[col].min(), df[col].max(), df[col].mean()) for col in df.columns]

    def calculate_scores_for_row(self, row):
    # Apply scoring only on relevant columns (from 3rd column onward)
        # range(2, len(row)) is used to skip the first two columns (x and y)
        # [j-2] is used to get the correct score object for the column because the first two columns are skipped aka x, y
        return [self.scores[j - 2].calculate_score(row[j]) for j in range(2, len(row))]


    def filtering(self, dataframe, filterFile):
        """
        Filter the dataframe and calculate scores for each row.
        """
        self.filter(filterFile)  # No need to store the result since it initializes self.scores

        # Apply score calculation to each row in the dataframe and expand results into new columns
        score_columns = dataframe.apply(self.calculate_scores_for_row, axis=1, result_type='expand')

        # Assign score columns to the dataframe
        dataframe[[f'score_{i}' for i in range(score_columns.shape[1])]] = score_columns

        return dataframe
    
    def total_score(self, dataframe):
        """
        Calculate the total score for the dataframe.
        """
        # add a new column to the dataframe that contains the sum of all score columns divided by the number of score columns
        num_score_cols = sum('score_' in col for col in dataframe.columns)
        
        dataframe['total_score'] = dataframe[[f'score_{i}' for i in range(1, num_score_cols)]].sum(axis=1) / num_score_cols

        return dataframe
