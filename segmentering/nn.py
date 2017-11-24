import numpy as np
import h5py
import sys
import sklearn
from sklearn.neighbors import KDTree
import pdb
import networkx 
from networkx.algorithms.components.connected import connected_components
import matplotlib.pyplot as plt

inputFile = open(sys.argv[1],"r")
pointCloud = []
groundLevel = -1.64
distance = sys.argv[2]
cluster = []
clusters = []

for line in inputFile:
	try:
		floats=[float(x) for x in line.split(" ")]
		if floats[2]>groundLevel:
			pointCloud.append(floats)

	except Exception as e:
		pass
tree = KDTree(pointCloud)


neighborhoods = tree.query_radius(pointCloud,r=distance)
for array in neighborhoods:
	clusters.append(array.tolist())



#print(clusters)

def to_graph(neighbors_clusters):
	G=networkx.Graph()
	for neighbors in neighbors_clusters:
		G.add_nodes_from(neighbors)
		G.add_edges_from(to_edges(neighbors))
	return G
	
def to_edges(neighbors):
	it = iter(neighbors)
	last = next(it)
	for current in it:
		yield last,current
		last = current
	
G=to_graph(clusters)
print connected_components(G)
#print G.neighbors(0)
print(len(G))
G.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                       node_color = values, node_size = 500)

#for neighbors_point_1 in neighborhoods:
#	cluster = neighbors_point_1
#	for n in range(len(neighborhoods)):
#		neighbors_point_2 = neighborhoods[n]
#		commonPoints = list(set(cluster).intersection(neighbors_point_2))
#		pdb.set_trace() 
#		if len(commonPoints)>0:
#			np.append(cluster,list(set(neighbors_point_2)-set(commonPoints)))
#			np.delete(neighborhoods,neighborhoods[n],None)
#			pdb.set_trace()
#	clusters.append(cluster)
	#print(cluster)
#print(len(clusters))
#print(len(neighborhoods))



#if (point[0] - neighbor[0]) < distance and (point[1] - neighbor[1]) < distance and (point[2] - neighbor[2]) < distance:
#	cluster.append(neighbor)
#for member in cluster:
#	if (point[0] - member[0]) < distance and (point[1] - member[1]) < distance and (point[2] - member[2]) < distance:
#		cluster.append(point)