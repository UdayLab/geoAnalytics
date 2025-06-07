
import pandas as pd
from geoAnalytics.conversion import DF2DB as df2db

class rasterDF2DB:

    def __init__(self,   dataframe):
        self.transactionDF = None
        self.df = dataframe


    def run(self):
        data = self.df.drop([0, 1], axis=1)
        newDF = pd.DataFrame()
        newDF['new_col'] = 'POINT(' + self.df[0].astype(str) + ',' + self.df[1].astype(str) + ')'
        self.transactionDF  = pd.concat([newDF, data.reset_index(drop=True)], axis=1).set_index(newDF.columns[0]).T
        print("Prepared transactional DataFrame:", self.transactionDF.shape)

    def convertToTransactionDB(self,oFile: str, condition: str, thresholdValue: Union[int, float]):
        obj = df2db.DF2DB(self.transactionDF)
        obj.convert2TransactionalDatabase(oFile, condition, thresholdValue)

    def convertToTemporalDB(self,oFile: str, condition: str, thresholdValue: Union[int, float]):
        obj = df2db.DF2DB(self.transactionDF)
        obj.convert2TemporalDatabase(oFile, condition, thresholdValue)