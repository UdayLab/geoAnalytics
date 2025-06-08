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
from sklearn.cluster import KMeans

class OverlapScore:
    def __init__(self, TrainDF, TopkDF, startBandTrainDF = 2, startBandTopkDF = 2):
        self.TrainDF = TrainDF.iloc[:, startBandTrainDF:]
        self.TopkDF = TopkDF.iloc[:, startBandTopkDF:]

        if self.TrainDF.shape[1] != self.TopkDF.shape[1]:
            raise ValueError("TrainDF and TopkDF must have the same number of columns after slicing.")

    def run(self, n_clusters=2):
        combined = np.vstack([self.TrainDF, self.TopkDF])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(combined)

        trainCluster = labels[:len(self.TrainDF)]
        topkCluster = labels[len(self.TrainDF):]

        overlapScore = np.sum(trainCluster[0] == topkCluster) / len(topkCluster)
        return overlapScore
