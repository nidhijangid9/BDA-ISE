# tracker.py
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import time

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        # next unique object ID
        self.nextObjectID = 0
        # dict: objectID -> centroid (x, y)
        self.objects = OrderedDict()
        # objectID -> number of consecutive frames it has disappeared
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        # store timestamps for bookkeeping
        self.last_seen = {}

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.last_seen[self.nextObjectID] = time.time()
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.last_seen[objectID]

    def update(self, rects):
        """
        rects: list of bounding boxes [(startX, startY, endX, endY), ...]
        returns: dict of objectID -> centroid
        """
        # if no detections, mark disappeared
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        # compute input centroids
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if no tracked objects yet, register all input centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(tuple(inputCentroids[i]))
        else:
            # build arrays of current object centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # find smallest distance pairings
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = tuple(inputCentroids[col])
                self.disappeared[objectID] = 0
                self.last_seen[objectID] = time.time()

                usedRows.add(row)
                usedCols.add(col)

            # compute unused rows/cols
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # if number of object centroids >= number of input centroids
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(tuple(inputCentroids[col]))

        return self.objects
