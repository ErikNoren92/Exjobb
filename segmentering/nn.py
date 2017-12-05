import numpy as np
import h5py
import sys
import sklearn
from sklearn.neighbors import KDTree
import pdb
import networkx as nx
from networkx.algorithms.components.connected import connected_components
import matplotlib.pyplot as plt
import random
from vispy import scene, visuals, app, gloo, io
from itertools import cycle

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
				pointCloud.append(floats[:3])
		except Exception as e:
			pass
	return pointCloud


def getNeighbors(tree,pointCloud):
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

def formatData(pointClusters,pointCloud):
	formatedClusters = []
	labels = []
	for points in pointClusters:
		if len(points) >= 128:
			formatedClusters.append(resize(points,pointCloud))
			labels.append(0)
		elif 128 > len(points) and len(points) > 20:
			formatedClusters.append(resize(points,pointCloud))
			labels.append(0)
		else:
			pass	
	return (formatedClusters, labels)

def resize(model,pointCloud):
	if len(model) > 128:
		for x in range(int(len(model)/pointLimit)):
			newSet = random.sample(model,pointLimit)
	else:
		newSet = model
	formatedCloud = []
	for point in newSet:
		formatedCloud.append(pointCloud[point])

	return formatedCloud

def retrieveData():
	pointCloud = readFile("set2.pcd")
	tree = KDTree(pointCloud)
	Graph=to_graph(getNeighbors(tree,pointCloud))
	subGraph = list(nx.connected_components(Graph))
	return formatData(subGraph,pointCloud)




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

data,labels = retrieveData()

array = np.asarray(([[0,0,0]]),dtype=np.float32)

b= np.asarray(data[0],dtype=np.float32)
#array = np.concatenate((array,(np.asarray(data[0],dtype=np.float32))),axis=0)
for x in range(len(data)):
	temp=np.asarray(data[x],dtype=np.float32)
	array = np.concatenate((array,temp),axis=0)
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

fov = 60.
cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov)
cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov)
cam3 = scene.cameras.ArcballCamera(parent=view.scene, fov=fov)

view.camera = cam2

# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    global opaque_cmap, translucent_cmap
    if event.text == '1':
        cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
        view.camera = cam_toggle.get(view.camera, 'fly')
    elif event.text == '2':
        methods = ['mip', 'translucent', 'iso', 'additive']
        method = methods[(methods.index(volume1.method) + 1) % 4]
        print("Volume render method: %s" % method)
        cmap = opaque_cmap if method in ['mip', 'iso'] else translucent_cmap
        volume1.method = method
        volume1.cmap = cmap
        volume2.method = method
        volume2.cmap = cmap
    elif event.text == '3':
        volume1.visible = not volume1.visible
        volume2.visible = not volume1.visible
    elif event.text == '4':
        if volume1.method in ['mip', 'iso']:
            cmap = opaque_cmap = next(opaque_cmaps)
        else:
            cmap = translucent_cmap = next(translucent_cmaps)
        volume1.cmap = cmap
        volume2.cmap = cmap
    elif event.text == '0':
        cam1.set_range()
        cam3.set_range()
    elif event.text != '' and event.text in '[]':
        s = -0.025 if event.text == '[' else 0.025
        volume1.threshold += s
        volume2.threshold += s
        th = volume1.threshold if volume1.visible else volume2.threshold
        print("Isosurface threshold: %0.3f" % th)

#for object in data:
#	print(object.shape)
#	print(object[0].shape)
color1 = np.array([[1, 1, 0]] * 9676)
#color2 = np.array([[0.4, 1, 0]] * 128)
#color3 = np.array([[0, 0.4, 1]] * 128)

print(color1.shape)

# Create the scatter plot
scatter = scene.visuals.Markers()
scatter.set_data(array, face_color=color1,size=1)
view.add(scatter)
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range()
app.run()