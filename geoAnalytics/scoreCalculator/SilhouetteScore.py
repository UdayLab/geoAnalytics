import numpy as np
from sklearn.metrics import silhouette_score

class SilhouetteScore:
    def __init__(self, TrainDF, TopkDF, startBandTrainDF, startBandTopkDF):
        self.TrainDF = TrainDF.iloc[:, startBandTrainDF:]
        self.TopkDF = TopkDF.iloc[:, startBandTopkDF:]

        if self.TrainDF.shape[1] != self.TopkDF.shape[1]:
            raise ValueError("TrainDF and TopkDF must have the same number of columns after slicing.")

    def run(self):
        combined = np.vstack([self.TrainDF, self.TopkDF])
        labels = np.array([1]*len(self.TrainDF) + [0]*len(self.TopkDF))
        score = silhouette_score(combined, labels)
        return score
