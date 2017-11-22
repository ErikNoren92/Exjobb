import numpy as np
import h5py
import sys
import sklearn
from scipy.spatial import KDTree

inputFile = open(sys.argv[1],"r")
pointCloud = []
groundLevel = -1.64
distance = 2
cluster = []
for line in inputFile:
	try:
		floats=[float(x) for x in line.split(" ")]
		if floats[2]>groundLevel:
			pointCloud.append(floats)

	except Exception as e:
		pass
tree = KDTree(pointCloud)






for point in pointCloud:
	point.append(cluster)
	pointCloud.remove(point)
	neighbors = tree.query_radius(point,radius=1)
	print(neighbors)




#if (point[0] - neighbor[0]) < distance and (point[1] - neighbor[1]) < distance and (point[2] - neighbor[2]) < distance:
#	cluster.append(neighbor)
#for member in cluster:
#	if (point[0] - member[0]) < distance and (point[1] - member[1]) < distance and (point[2] - member[2]) < distance:
#		cluster.append(point)