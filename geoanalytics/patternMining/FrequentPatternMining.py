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
from PAMI.extras.dbStats.TransactionalDatabase import TransactionalDatabase
from PAMI.frequentPattern.basic import FPGrowth

class FrequentPatternMining:
    def __init__(self, inputFile=''):
        self.inputFile = inputFile
        self.miner = None  # will hold the FPGrowth instance

    def showDBstats(self):
        obj = TransactionalDatabase(self.inputFile)
        obj.run()
        obj.printStats()
        obj.plotGraphs()

    def run(self, minSupport=8):
        self.miner = FPGrowth.FPGrowth(self.inputFile, minSupport)
        self.miner.startMine()
        self.miner.printResults()

    def save(self, outputFile='FrequentPatterns.txt'):
        if self.miner is not None:
            self.miner.save(outputFile)
            print(f"Frequent patterns saved to: {outputFile}")
        else:
            print("No mining results to save. Please execute run() method first.")
