# findNeighboursUsingEuclidean is a code used to create a neighbourhood file using Euclidean distance.
#
#  **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.neighbours import findNeighboursUsingEuclidean as db
#
#     obj = db.findNeighboursUsingGeodesic(iFile, oFile, 10, "\t")
#
#     obj.save()
#


__copyright__ = """
Copyright (C)  2021 Rage Uday Kiran

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

import re
from math import sqrt
import time
import sys, psutil, os,tqdm
import pandas as pd
import numpy as np
from tqdm import tqdm


class FindNeighboursUsingEuclidean:
    """
    This class create a neighbourhood file using euclid distance.

    :Attribute:

        :param iFile : file
            Input file name or path of the input file
        :param maxDist : int
            The user can specify maxEuclideanDistance.
            This program find pairs of values whose Euclidean distance is less than or equal to maxEucledianDistace
            and store the pairs.
        :param  sep: str :
                    This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.

    :Methods:

        mine()
            find and store the pairs of values whose Euclidean distance is less than or equal to maxEucledianDistace.
        getFileName()
            This function returns output file name.

    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.neighbours import findNeighboursUsingEuclidean as db

            obj = db.findNeighboursUsingEuclidean(iFile, oFile, 10, "\t")

            obj.save()
    """

    def __init__(self, iFile: str, maxDist: int, sep='\t', DBtype="csv"):
        self.iFile = iFile
        self.maxEucledianDistance = maxDist
        self.seperator = sep
        self.DBtype = DBtype

    def createAndSave(self, oFile: str):
        # Load coordinates
        if self.DBtype == "csv":
            df = pd.read_csv(self.iFile)
            coords = df.iloc[:, [0, 1]].astype(float).values
        else:
            coords = []
            seen = set()
            with open(self.iFile, "r") as f:
                for line in f:
                    parts = line.rstrip().split(self.seperator)
                    for part in parts[1:]:
                        cleaned = re.sub(r'[^0-9. ]', '', part).strip()
                        if cleaned and cleaned not in seen:
                            seen.add(cleaned)
                            try:
                                x, y = map(float, cleaned.split())
                                coords.append([x, y])
                            except ValueError:
                                continue
            # Convert to array
            # coords = cp.array(coords)
            coords = np.array(coords)

        if self.DBtype == "csv":
            coords = np.asarray(coords)

        if coords.shape[0] == 0:
            return

        # Compute distances on the GPU using broadcasting
        # This creates the full distance matrix, but we will not pull all of it to CPU
        # dists = cp.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        # Create a boolean mask for neighbors within given distance excluding self (distance==0)
        within_dist = (dists <= self.maxEucledianDistance) & (dists > 0)

        print(f"Number of points: {coords.shape[0]}")
        print(f"Number of neighbors: {within_dist.sum()}")

        # Open file for direct writing
        with open(oFile, "w") as f:
            # Process each row individually to keep the CPU memory footprint low.
            for i in tqdm(range(coords.shape[0])):
                # print(f"Processing point {i+1}/{coords.shape[0]}")
                # Transfer only the i-th point and its corresponding neighbor mask to CPU
                # point = cp.asnumpy(coords[i])
                point = coords[i]
                # neighbor_mask = cp.asnumpy(within_dist[i])
                neighbor_mask = within_dist[i]
                if neighbor_mask.any():
                    # Write the reference point
                    line = f"Point({point[0]}, {point[1]})"
                    # Transfer only the neighbor points for the true mask.
                    # neighbors = cp.asnumpy(coords[neighbor_mask])
                    neighbors = coords[neighbor_mask]
                    # Append each neighbor
                    for neighbor in neighbors:
                        # line += f"\tPoint({neighbor[0]} {neighbor[1]})"
                        line += f"\tPoint({int(neighbor[0])}, {int(neighbor[1])})"
                    f.write(line + "\n")

    def getNeighboringInformation(self):
        df = pd.DataFrame(['\t'.join(map(str, line)) for line in self.result], columns=['Neighbors'])
        return df

    def getRuntime(self) -> float:
        """
        Get the runtime of the transactional database

        :return: the runtime of the transactional database


        :rtype: float
        """
        return self._endTime - self._startTime

    def getMemoryUSS(self) -> float:

        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        return self._memoryUSS

    def getMemoryRSS(self) -> float:

        process = psutil.Process(os.getpid())
        self._memoryRSS = process.memory_info().rss
        return self._memoryRSS


if __name__ == "__main__":
    obj = FindNeighboursUsingEuclidean(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
