import numpy as np
from sklearn.cluster import KMeans

class OverlapScore:
    def __init__(self, TrainDF, TopkDF, startBandTrainDF, startBandTopkDF):
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
