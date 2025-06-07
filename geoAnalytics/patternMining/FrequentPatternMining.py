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
from PAMI.extras.convert import DF2DB
from PAMI.extras.dbStats.TransactionalDatabase import TransactionalDatabase
from PAMI.frequentPattern.basic import FPGrowth

class FrequentPatternMining:
    def __init__(self, dataframe, outputTransactionPath='transactionDB.csv'):
        self.df = dataframe.copy()
        self.outputTransactionPath = outputTransactionPath
        self.transactionDF = None

    def prepareTransactionalDataframe(self):
        data = self.df.drop([0, 1], axis=1)
        newDF = pd.DataFrame()
        newDF['new_col'] = 'POINT(' + self.df[0].astype(str) + ',' + self.df[1].astype(str) + ')'
        newDF = pd.concat([newDF, data.reset_index(drop=True)], axis=1)
        newDF = newDF.set_index(newDF.columns[0]).T
        self.transactionDF = newDF
        print("Prepared transactional DataFrame:", self.transactionDF.shape)

    def convertToTransactionDB(self, condition='>=', thresholdValue=4000):
        obj = DF2DB.DF2DB(self.transactionDF)
        obj.convert2TransactionalDatabase(
            oFile=self.outputTransactionPath,
            condition=condition,
            thresholdValue=thresholdValue
        )
        print(f"Saved transaction DB to: {self.outputTransactionPath}")

    def showDBstats(self):
        obj = TransactionalDatabase(self.outputTransactionPath)
        obj.run()
        obj.printStats()
        obj.plotGraphs()

    def run(self, minSupport=8, outputFile='FrequentPatterns.txt'):
        obj = FPGrowth.FPGrowth(self.outputTransactionPath, minSupport)
        obj.startMine()
        obj.printResults()
        obj.save(outputFile)
        print(f"Frequent patterns saved to: {outputFile}")