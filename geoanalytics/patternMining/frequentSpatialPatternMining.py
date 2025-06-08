

#input parameters for the class

      #rasterDataframe, maxtDist, neighhborhoodFileName, distanceMeasure="geodic (default) or euclidean", condition, thresholdValues, and minimumSupport
      #save(outputPatternFile)








#-------------------------
#Extract Data
#data = self.df.drop(['x', 'y'], axis=1)

# Concatenate X and Y columns of the dataframe using a tab space
#newDF['new_col'] = 'POINT(' + df['col1'].astype(str) + ',' + df['col2'].astype(str) + ')'


# Join new dataframe with the data dataframe
#newDF = pd.concat([newDF, data.reset_index(drop=True)], axis=1)

#Save the new dataframe as a file, say newDF.txt

# Finding neighbors using Eucledian or Geodic distances
##from geoanalytics.extras.neighbours import FindNeighboursUsingEuclidean as eucl
from geoanalytics.extras.neighbours import  FindNeighboursUsingGeodesic as geodesic

obj = geodesic.FindNeighboursUsingGeodesic('newDF.txt',100,DBtype='trans')

obj.create()

obj.save('neighbors.txt')


# Execute the algorithm frequentSpatialMiningAlgorithm with inputs (newDF.txt, neighbors.txt, minSup=100)

# from PAMI.georeferencedFrequentPattern.basic import FSPGrowth as alg
#
#             obj = alg.FSPGrowth("sampleTDB.txt", "sampleN.txt", 5)
#
#             obj.mine()
#
#             spatialFrequentPatterns = obj.getPatterns()
#
#             print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))
#
#             obj.save("outFile")
#
#             memUSS = obj.getMemoryUSS()
#
#             print("Total Memory in USS:", memUSS)
#
#             memRSS = obj.getMemoryRSS()
#
#             print("Total Memory in RSS", memRSS)
#
#             run = obj.getRuntime()
#
#             print("Total ExecutionTime in seconds:", run)


