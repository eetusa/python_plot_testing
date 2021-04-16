import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

center = (0,0)
radius = 1.0

p1 = [0,0]


def getRandomStepDistance():
	return random.uniform(0.001, 0.1)

def getRandomDirection():
	return random.uniform(0, 2 * math.pi)

def isPointTooFarFromCenter(point):
	return getDistance(point, center) > radius

def getDistance(point1, point2):
	return math.sqrt( pow(point2[0] - point1[0],2) + pow(point2[1] - point1[1], 2))

def movePoint(point):
	temp = [-0.5,-0.5]

	temp[0] = point[0]
	temp[1] = point[1]

	direction = getRandomDirection()
	stepdistance = getRandomStepDistance()

	point[0] = round(point[0] + math.cos(direction)*stepdistance, 4)
	point[1] = round(point[1] + math.sin(direction)*stepdistance, 4)

	if (isPointTooFarFromCenter(point)):
		point[0] = temp[0]
		point[1] = temp[1]
		movePoint(point)

def generatePointsFromCenter(n):
	if n < 1: 
		return
	point = [0, 0]
	arr = np.empty((0,2))
	arr = np.append(arr, np.array([point]), axis=0)
	
	for i in range(n-1):
		movePoint(point)
		arr = np.append(arr, np.array([point]), axis=0)
		
	return arr


arr = generatePointsFromCenter(1000)
print(arr.shape)


#print(arr)


circle2 = plt.Circle((0,0), 1, color='r', fill=False)


plt.figure(figsize=(10,10))

plt.plot(*zip(*arr), lw=0.3, marker='', color='r')

#fig, ax = plt.subplots()
plt.gca().add_patch(circle2)

df = pd.DataFrame(arr, columns = ['x', 'y'])
print(df.describe())
#a = sns.scatterplot(x=df.x, y=df.y)
#sns.scatterplot(x=df.x, y=df.y, data=df)
#a.ax_joint.plot([0],[0],'o',ms=10 , mec='r', mfc='none')
plt.show()