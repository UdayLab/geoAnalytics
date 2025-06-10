import pandas as pd
from geoanalytics.conversion import DF2DB

class RasterDF2DB:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = ['x', 'y'] + [str(i) for i in range(1, self.df.shape[1] - 1)]
        self.transactionDF = None

    def prepareTransactionalDataframe(self):
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
        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2TransactionalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved transaction DB to: {DBname}")

    def convertToTemporalDB(self, DBname='temporalDB.csv', condition='>=', thresholdValue=4000):
        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2TemporalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved temporal DB to: {DBname}")

    def convertToUtilityDB(self, DBname='UtilityDB.csv'):
        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2UtilityDatabase(
            oFile=DBname
        )
        print(f"Saved utility DB to: {DBname}")

    def convertToGeoReferencedTransactionalDB(self, DBname='geoReferencedTransactionalDatabase.csv', condition='>=', thresholdValue=4000):
        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2geoReferencedTransactionalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved geo-referenced transaction DB to: {DBname}")

    def convertToGeoReferencedTemporalDB(self, DBname='geoReferencedTemporalDatabase.csv', condition='>=', thresholdValue=4000):
        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2geoReferencedTemporalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved geo-referenced temporal DB to: {DBname}")

    def convertToUncertainTransactionalDB(self, DBname='UncertainTransactionalDB.csv', condition='>=', thresholdValue=4000):
        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2UncertainTransactionalDatabase(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved uncertain transaction DB to: {DBname}")

    def convertToMultipleTimeSeries(self, DBname='MultipleTimeSeriesDB.csv', condition='>=', thresholdValue=4000, interval = 2):
        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2MultipleTimeSeries(
            oFile=DBname,
            condition=condition,
            thresholdValue=thresholdValue,
            interval=interval
        )
        print(f"Saved multiple time series DB to: {DBname}")