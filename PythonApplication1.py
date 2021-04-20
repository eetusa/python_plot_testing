import math
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from seaborn.matrix import _HeatMapper
from seaborn.external.six import string_types

from seaborn.utils import despine, axis_ticklabels_overlap, relative_luminance, to_utf8
import time


def insertCenterEveryOther(arrayz):
	arr = np.empty((0,2))
	print(arrayz.shape)
	print(arr.shape)

	point = [0,0]

	for i in range (len(arrayz)):
		arr = np.append(arr, np.array([point]), axis=0)
		arr = np.append(arr, np.array([arrayz[i]]), axis=0)

	return arr



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

def movePointV2(point):
	temp = [-0.5,-0.5]

	temp[0] = point[0]
	temp[1] = point[1]

	deltax = random.uniform(-0.1,0.1)
	deltay = random.uniform(-0.1,0.1)

	point[0] = round(point[0] + deltax, 4)
	point[1] = round(point[1] + deltay, 4)

	if (isPointTooFarFromCenter(point)):
		point[0] = temp[0]
		point[1] = temp[1]
		movePointV2(point)

def generatePointsFromCenter(n):
	if n < 1: 
		return
	point = [0.9,0]
	arr = np.empty((0,2))
	arr = np.append(arr, np.array([point]), axis=0)
	
	for i in range(n-1):
		movePointV2(point)
		arr = np.append(arr, np.array([point]), axis=0)
		
	return arr

def getPointsInSectorAboveArm(sectorArmsVectors, startArmInt, array):
	if (sectorArmsVectors.shape[0] <= startArmInt): return
	if (startArmInt < 0): return

	startArm = sectorArmsVectors[startArmInt]
	endArm = sectorArmsVectors[0] if startArmInt==sectorArmsVectors.shape[0]-1 else sectorArmsVectors[startArmInt+1]

	arr = np.empty((0,2))
	arrSize = array.shape[0]

	for i in range( arrSize ):
		if (isPointBetweenArms(startArm, endArm, array[i])):
			arr = np.append(arr, np.array([array[i]]), axis=0)

	return arr

#def isPointBetweenArms(arm1, arm2, point):
	#return ((point[0]*arm1[0] + point[1]*arm1[1]) > 0 and (point[0]*arm2[0] + point[1]*arm2[1]) < 0)
	
def isPointBetweenArms(arm1, arm2, pointOr):
	point = [-pointOr[1], pointOr[0]]
	return ((point[0]*arm1[0] + point[1]*arm1[1]) < 0 and (point[0]*arm2[0] + point[1]*arm2[1]) > 0)

def getPointsInWidthAndRadius(array, subradius, width):
	arr = np.empty((0,2))
	arrSize = array.shape[0]
	center_point = [0, 0]
	upperlimit = subradius+width

	for i in range(arrSize):
		distance = getDistance(array[i], center_point)
		if (distance >= subradius and distance < upperlimit):
			arr = np.append(arr, np.array([array[i]]), axis=0)

	return arr

def getAllSubSectorsInSector(sector_array, circleRadius, sub_sec_amount, biggest_amount, smallest_amount):
	whole_sector_array = []

	stepWidth = circleRadius / sub_sec_amount

	for i in range(sub_sec_amount):
		subsector = getPointsInWidthAndRadius(sector_array, i*stepWidth, stepWidth)
		if (subsector.shape[0] > biggest_amount):
			biggest_amount = subsector.shape[0]
		if (subsector.shape[1] < smallest_amount):
			smallest_amount = subsector.shape[1]
		print(subsector.shape)
		whole_sector_array.append(subsector)

	return whole_sector_array
	


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def plotSubSectors(allSubSecsArray, biggest_amount, smallest_amount):
	#c1='#1f77b4' #blue
	c2='red' #blue
	c1='blue' #green
	total_element_count = 0
	#biggest_amount = -float('inf')
	#smallest_amount = float('inf')

	for i in range(len(allSubSecsArray)):
		total_element_count += allSubSecsArray[i].shape[0]
		if (allSubSecsArray[i].shape[0] > biggest_amount):
			biggest_amount = allSubSecsArray[i].shape[0]

		if (allSubSecsArray[i].shape[0] < smallest_amount):
			smallest_amount = allSubSecsArray[i].shape[0]

	for i in range(len(allSubSecsArray)):
		if (biggest_amount==smallest_amount):
			plt.plot(*zip(*allSubSecsArray[i]), lw=0, marker='x', color=colorFader(c1,c2,(allSubSecsArray[i].shape[0]-smallest_amount)/(0.01))) ##change to relative of biggest
		else:
			plt.plot(*zip(*allSubSecsArray[i]), lw=0, marker='x', color=colorFader(c1,c2,(allSubSecsArray[i].shape[0]-smallest_amount)/(biggest_amount-smallest_amount))) ##change to relative of biggest

def getSectorArmVectors(n, startingDegree):
	arms_array = np.empty((0,2))
	
	stepSize = 2*math.pi/n
	print("start: ",startingDegree)
	r = startingDegree * (math.pi/180)
	print("r: ",r)
	print("stepsize: ",stepSize)
	for i in range(n):
		rd = r + i * stepSize
		if (rd > 2*math.pi):
			rd = rd - 2*math.pi
		print("rad: ",rd,n,stepSize)
		point = [round(math.cos(rd),5), round(math.sin(rd),5)]
		arms_array = np.append(arms_array, np.array([point]), axis=0)

	return arms_array

def getAllSectors(sector_arms, all_data, sub_sector_amount, biggest_amount, smallest_amount):
	all_sectors = []
	for i in range(sector_arms.shape[0]):
		sector_points = getPointsInSectorAboveArm(sector_arms,i, arr1)
		sector_divided = getAllSubSectorsInSector(sector_points, radius, sub_sector_amount, biggest_amount, smallest_amount)
		all_sectors.append(sector_divided)

	return all_sectors

def getAndDrawSectors(pointdata, sector_amount, subsector_amount):
	biggest_amount = -float('inf')
	smallest_amount = float('inf')

	sectorArmsVectors = getSectorArmVectors(sector_amount, 45)
	sectorArmsDrawVectors = insertCenterEveryOther(sectorArmsVectors)
	#plt.plot(*zip(*sectorArmsDrawVectors), lw=0.9, marker='', color='b')

	all_divided = getAllSectors(sectorArmsVectors, pointdata, subsector_amount, biggest_amount, smallest_amount)
	print("here")
	for i in range(len(all_divided)):
		for  j in range (len(all_divided[i])):
			amount = all_divided[i][j].shape[0]
			if (amount > biggest_amount):
				biggest_amount = amount
			if (amount < smallest_amount and amount != 0):
				smallest_amount = amount

	for i in range(len(all_divided)):
		plotSubSectors(all_divided[i], biggest_amount, smallest_amount)




center = (0,0)
radius = 1.0

p1 = [0,0]
amountOfDataPoints = 10000

###circle2 = plt.Circle((0,0), 1, color='r', fill=False)
plt.figure(figsize=(10,10))

#sectorArmsVectors = getSectorArmVectors(16, 45)


#sectorArmsDrawVectors = insertCenterEveryOther(sectorArmsVectors)




start = time.process_time()
#arr_sector = getPointsInSectorAboveArm(3, arr1)


#all_divided = getAllSectors(sectorArmsVectors, arr1, 8)

#arr_divided = getAllSubSectorsInSector(getPointsInSectorAboveArm(sectorArmsVectors,3, arr1), radius, 8)
arr1 = generatePointsFromCenter(amountOfDataPoints)


print(arr1.shape)


elapsed = (time.process_time() - start)
print("timed " + str(elapsed))

#ax = sns.heatmap(*zip(*arr1))

#for i in range(len(all_divided)):
#	plotSubSectors(all_divided[i])
#plotSubSectors(arr_divided)



getAndDrawSectors(arr1, 64, 64)
#plt.plot(*zip(*arr1), lw=0.2, marker='', color='r')
#plt.plot(*zip(*arr), lw=0, marker='o', color='r')

#fig, ax = plt.subplots()
###plt.gca().add_patch(circle2)
df = pd.DataFrame(arr1, columns = ['x', 'y'])
print(df.describe())
#a = sns.scatterplot(x=df.x, y=df.y)
#sns.scatterplot(x=x, y=y, data=arr)
#a.ax_joint.plot([0],[0],'o',ms=10 , mec='r', mfc='none')
plt.show()