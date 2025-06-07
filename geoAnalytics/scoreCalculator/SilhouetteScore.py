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

import numpy as np
from sklearn.metrics import silhouette_score

class SilhouetteScore:
    def __init__(self, TrainDF, TopkDF, startBandTrainDF = 2, startBandTopkDF = 2):
        self.TrainDF = TrainDF.iloc[:, startBandTrainDF:]
        self.TopkDF = TopkDF.iloc[:, startBandTopkDF:]

        if self.TrainDF.shape[1] != self.TopkDF.shape[1]:
            raise ValueError("TrainDF and TopkDF must have the same number of columns after slicing.")

    def run(self):
        combined = np.vstack([self.TrainDF, self.TopkDF])
        labels = np.array([1]*len(self.TrainDF) + [0]*len(self.TopkDF))
        score = silhouette_score(combined, labels)
        return score
