import numpy as np
import h5py
import sys
import sklearn
from sklearn.neighbors import KDTree
import pdb
import networkx as nx
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
	G=nx.Graph()
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

sub = list(nx.connected_components(G))
#print s[0].nodes()
#print nx.number_connected_components(G)
print type(sub[0])
print list(sub[1])
fileName = 0
for cluster in sub:
	list(cluster)
	fileName += 1
	output=open((str(fileName)+".pcd"),"w")
	output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(cluster)) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(cluster))+"\nDATA ascii\n")
	for node in cluster:
		output.write(str(pointCloud[node][0])+" "+str(pointCloud[node][1])+" "+str(pointCloud[node][2])+"\n")
	output.close()
#print s[0].nodes()
nx.write_graphml(G,"test.graphml")



#                       node_color = values, node_size = 500)

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