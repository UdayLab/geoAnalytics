import numpy as np
from scipy.spatial.distance import cdist

class CrossDistance:
    def __init__(self, TopkDF, TrainDF, startBandTopkDF, startBandTrainDF):
        self.TopkDF = TopkDF.iloc[:,startBandTopkDF:]
        self.TrainDF = TrainDF.iloc[:,startBandTrainDF:]

        if self.TopkDF.shape[1] != self.TrainDF.shape[1]:
            raise ValueError("TopkDF and TrainDF must have the same number of columns after slicing.")

    def run(self, metric='euclidean'):
        distances = cdist(self.TopkDF, self.TrainDF, metric=metric)
        return np.mean(distances)