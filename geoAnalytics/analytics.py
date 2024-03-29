import pandas as pd
import numpy as np


def cosineSimilarity(tableName1, tableName2):
    """
    :Description: This function calculates the cosine similarity between two dataframes using the provided formula.
    Cosine Similarity = Dot product of df1 and df2 / (Magnitude of df1 * Magnitude of df2)

    :param tableName1: The first data frame containing 'x' and 'y' columns (point co-ordinates)
    :param tableName2: The second data frame containing 'x' and 'y' columns (point co-ordinates)

    :return: A new DataFrame with 'x' , 'y' and 'cosine similarity' columns
    :rtype: DataFrame
    """

    df1 = tableName1
    df2 = tableName2

    cosineSimilarityDF = pd.DataFrame(columns=['X', 'Y', 'Cosine Similarity'])

    # get x and y
    x = df1['x']
    y = df1['y']

    # drop x and y
    df1 = df1.drop(['x', 'y'], axis=1)
    df2 = df2.drop(['x', 'y'], axis=1)

    # convert to numpy array
    df1 = df1.to_numpy()
    df2 = df2.to_numpy()

    # calculate cosine similarity

    # for each row in df2
    for i in range(len(df2)):
        similarity_scores = df1.dot(df2) / (np.linalg.norm(df1, axis=1) * np.linalg.norm(df2))
        # get average of similarity scores
        average_similarity_score = np.average(similarity_scores)
        # append to dataframe
        cosineSimilarityDF = cosineSimilarityDF.append(
            {'X': x[i], 'Y': y[i], 'Cosine Similarity': average_similarity_score}, ignore_index=True)

    return cosineSimilarityDF

