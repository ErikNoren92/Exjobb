import numpy as np
import h5py
import sys
import sklearn
from sklearn.neighbors import KDTree
import pdb
import networkx as nx
from networkx.algorithms.components.connected import connected_components
import matplotlib.pyplot as plt


groundLevel = -1.64
distance = 0.5
pointLimit = 128

def readFile(fileName):
	pointCloud = []
	inputFile = open(fileName,"r")
	for line in inputFile:
		try:
			floats=[float(x) for x in line.split(" ")]
			if floats[2]>groundLevel and floats[0] < 40 and floats[0] > -40 and floats[1] < 40 and floats[1] > -40 :
				pointCloud.append(floats)
		except Exception as e:
			pass
	return pointCloud


def getNeighbors():
	clusters = []
	neighborhoods = tree.query_radius(pointCloud,r=distance)
	for array in neighborhoods:
		clusters.append(array.tolist())
	return clusters

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

def formatData(pointClusters):
	formatedClusters = []
	labels = []
	for points in pointClusters:
		if len(points) >= 128:
			formatedClusters.append(resize(points))
			labels.append(0)	
	return (formatedClusters, labels)

def resize(model):
	for x in range(int(len(model)/pointLimit)):
		newSet = random.sample(model,pointLimit)
		return newSet

def retrieveData():
	pointCloud = readFile("set2.pcd")
	tree = KDTree(pointCloud)
	Graph=to_graph(getNeighbors())
	subGraph = list(nx.connected_components(Graph))
	return formatData(subGraph)


#fileName = 0
#for cluster in subGraph:
#	if len(cluster) > 30:	
#		list(cluster)
#		fileName += 1
#		output=open((str(fileName)+".pcd"),"w")
#		output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(cluster)) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(cluster))+"\nDATA ascii\n")
#		for node in cluster:
#			output.write(str(pointCloud[node][0])+" "+str(pointCloud[node][1])+" "+str(pointCloud[node][2])+"\n")
#		output.close()
#nx.write_graphml(Graph,"test.graphml")
