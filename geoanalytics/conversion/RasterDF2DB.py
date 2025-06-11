# RasterDF2DB converts a raster-based DataFrame (with coordinates and band values) into various types of structured databases for spatio-temporal data mining tasks.
#
# **Importing and Using the RasterDF2DB Class in a Python Program**
#
#         import pandas as pd
#
#         from geoanalytics.conversion import RasterDF2DB
#
#         df = pd.read_csv('output.csv')
#
#         converter = RasterDF2DB(dataframe=df)
#
#         converter.prepareTransactionalDataframe()
#
#         converter.convertToTransactionalDB(DBname='transactionDB.csv', condition='>=', thresholdValue=4000)
#
#         converter.convertToTemporalDB(DBname='temporalDB.csv', condition='>=', thresholdValue=4000)
#
#         converter.convertToUtilityDB(DBname='utilityDB.csv')
#
#         converter.convertToGeoReferencedTransactionalDB(DBname='geoTransactionDB.csv', condition='>=', thresholdValue=4000)
#
#         converter.convertToGeoReferencedTemporalDB(DBname='geoTemporalDB.csv', condition='>=', thresholdValue=4000)
#
#         converter.convertToUncertainTransactionalDB(DBname='uncertainDB.csv', condition='>=', thresholdValue=4000)
#
#         converter.convertToMultipleTimeSeries(DBname='multiTimeSeriesDB.csv', condition='>=', thresholdValue=4000, interval=2)
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

import pandas as pd
from geoanalytics.conversion import DF2DB

class RasterDF2DB:
    """
    **About this algorithm**

    :**Description**: RasterDF2DB is a converter that takes a raster-format DataFrame (x, y, bands) and transforms it into various spatio-temporal database formats using DF2DB methods.

    :**Parameters**:    - **dataframe** (*pandas.DataFrame*): Input DataFrame with columns [x, y, band1, band2, ..., bandN].

    :**Attributes**:    - **df** (*DataFrame*): A copy of the input DataFrame with renamed columns.
                        - **transactionDF** (*DataFrame*): Transposed DataFrame used for database conversion.

    **Execution methods**

    .. code-block:: python

            import pandas as pd

            from geoanalytics.conversion import RasterDF2DB

            df = pd.read_csv('output.csv')

            converter = RasterDF2DB(dataframe=df)

            converter.prepareTransactionalDataframe()

            converter.convertToTransactionalDB(DBname='transactionDB.csv', condition='>=', thresholdValue=4000)

            converter.convertToTemporalDB(DBname='temporalDB.csv', condition='>=', thresholdValue=4000)

            converter.convertToUtilityDB(DBname='utilityDB.csv')

            converter.convertToGeoReferencedTransactionalDB(DBname='geoTransactionDB.csv', condition='>=', thresholdValue=4000)

            converter.convertToGeoReferencedTemporalDB(DBname='geoTemporalDB.csv', condition='>=', thresholdValue=4000)

            converter.convertToUncertainTransactionalDB(DBname='uncertainDB.csv', condition='>=', thresholdValue=4000)

            converter.convertToMultipleTimeSeries(DBname='multiTimeSeriesDB.csv', condition='>=', thresholdValue=4000, interval=2)


    **Credits**

    This implementation was created and revised  under the guidance of Professor Rage Uday Kiran.
    """

    def __init__(self, dataframe):
        """
        Constructor to initialize the RasterDF2DB object.

        :param dataframe: A pandas DataFrame containing x, y coordinates and raster band values.
        """

        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + [str(i) for i in range(1, self.df.shape[1] - 1)]
        self.transactionDF = None

    def prepareTransactionalDataframe(self):
        """
        Prepares the transactional format of the DataFrame.

        Converts [x, y, band1, band2, ..., bandN] format into a transposed format where each spatial point becomes a column and rows represent time intervals.
        """

        data = self.df.iloc[:, 2:]
        point_labels = 'POINT(' + self.df.iloc[:, 0].astype(str) + ',' + self.df.iloc[:, 1].astype(str) + ')'
        newDF = pd.DataFrame()
        newDF['new_col'] = point_labels
        newDF = pd.concat([newDF, data.reset_index(drop=True)], axis=1)
        newDF = newDF.set_index('new_col').T
        newDF.index = newDF.index.astype(int)
        self.transactionDF = newDF
        print("Prepared transactional DataFrame:", self.transactionDF.shape)

    def convertToTransactionalDB(self, DBname='transactionalDB.csv', condition='>=', thresholdValue=4000):
        """
        Converts the transactional DataFrame into a basic transactional database format.

        :param DBname: Output file name for the database.
        :param condition: Filtering condition (e.g., '>=').
        :param thresholdValue: Value threshold to consider.
        """

        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2TransactionalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved transaction DB to: {DBname}")

    def convertToTemporalDB(self, DBname='temporalDB.csv', condition='>=', thresholdValue=4000):
        """
        Converts to a temporal database format.

        :param DBname: Output file name.
        :param condition: Filtering condition.
        :param thresholdValue: Threshold value.
        """

        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2TemporalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved temporal DB to: {DBname}")

    def convertToUtilityDB(self, DBname='UtilityDB.csv'):
        """
        Converts to a utility database format (with weights).

        :param DBname: Output file name.
        """

        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2UtilityDatabase(
            oFile=DBname
        )
        print(f"Saved utility DB to: {DBname}")

    def convertToGeoReferencedTransactionalDB(self, DBname='geoReferencedTransactionalDatabase.csv', condition='>=', thresholdValue=4000):
        """
        Converts to a geo-referenced transactional database format.

        :param DBname: Output file name.
        :param condition: Filtering condition.
        :param thresholdValue: Threshold value.
        """

        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2geoReferencedTransactionalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved geo-referenced transaction DB to: {DBname}")

    def convertToGeoReferencedTemporalDB(self, DBname='geoReferencedTemporalDatabase.csv', condition='>=', thresholdValue=4000):
        """
        Converts to a geo-referenced temporal database format.

        :param DBname: Output file name.
        :param condition: Filtering condition.
        :param thresholdValue: Threshold value.
        """

        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2geoReferencedTemporalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved geo-referenced temporal DB to: {DBname}")

    def convertToUncertainTransactionalDB(self, DBname='UncertainTransactionalDB.csv', condition='>=', thresholdValue=4000):
        """
        Converts to an uncertain transactional database format (for probabilistic scenarios).

        :param DBname: Output file name.
        :param condition: Filtering condition.
        :param thresholdValue: Threshold value.
        """

        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2UncertainTransactionalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved uncertain transaction DB to: {DBname}")

    def convertToMultipleTimeSeries(self, DBname='MultipleTimeSeriesDB.csv', condition='>=', thresholdValue=4000, interval = 2):
        """
        Converts data into multiple time series format.

        :param DBname: Output file name.
        :param condition: Filtering condition.
        :param thresholdValue: Threshold value.
        :param interval: Interval of time steps to be used.
        """

        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2MultipleTimeSeries(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue,
            interval=interval
        )
        print(f"Saved multiple time series DB to: {DBname}")