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
from scipy.spatial.distance import cdist

class CrossDistance:
    def __init__(self, TopkDF, TrainDF, startBandTopkDF = 2, startBandTrainDF = 2):
        self.TopkDF = TopkDF.iloc[:,startBandTopkDF:]
        self.TrainDF = TrainDF.iloc[:,startBandTrainDF:]

        if self.TopkDF.shape[1] != self.TrainDF.shape[1]:
            raise ValueError("TopkDF and TrainDF must have the same number of columns after slicing.")

    def run(self, metric='euclidean'):
        distances = cdist(self.TopkDF, self.TrainDF, metric=metric)
        return np.mean(distances)