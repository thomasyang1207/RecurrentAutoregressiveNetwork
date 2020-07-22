import os, sys, shutil

import tensorflow as tf

import xml.etree.ElementTree as ET

from config import Cfg

from collections import defaultdict

import numpy as np


# Take a series of files containing bounding box information, parse them into sequences of bounding boxes (numpy arrays). 

# Data Format: [x, y, w, h](current), [x, y, w, h](previous), [x, y, w, h] * 10, 

# bounding boxes not associated will be used to initialize NEW TRACKS 

class Detection(object):
	def __init__(self, x_, y_, w_, h_, ID_):
		self.x = x_
		self.y = y_
		self.w = w_
		self.h = h_
		self.ID = ID_


	def __repr__(self):
		return repr('[ ' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.w) + ', ' + str(self.h) + ']')




class TrainingVideo(object):
	def __init__(self, numFrames):
		self.TrackedObjects = defaultdict(lambda : [None] * numFrames)
		self.N = numFrames


	def AddDetection(self, detection, trackID, frame):
		#if trackID doesn't exist... 
		self.TrackedObjects[trackID][frame] = detection




def GetTrainingSets(trainPath):
	#From the root directory, start gathering video sets

	trainingSets = os.listdir(trainPath)
	return trainingSets
			



def GetVideos(setPath):
	videos = os.listdir(setPath)
	return videos



def ProcessDetections(xmlFile):
	# Process XML file...
	#print(xmlFile)
	tree = ET.parse(xmlFile)
	root = tree.getroot()

	detectionList = []

	size = root.find('size')

	w = int(size.find('width').text)
	h = int(size.find('height').text)

	for obj in root.findall('object'):
		trackID = int(obj.find('trackid').text)
		bbox = obj.find('bndbox')

		xmin = int(bbox.find('xmin').text)
		ymin = int(bbox.find('ymin').text)
		xmax = int(bbox.find('xmax').text)
		ymax = int(bbox.find('ymax').text)

		detection = Detection(xmin, ymin, xmax-xmin, ymax-ymin, trackID)
		detectionList.append(detection)


	return np.array(detectionList), w, h





def ProcessVideo(videoPath):
	frames = os.listdir(videoPath)
	frames.sort()
	#print(frames)
	numFrames = len(frames)
	trainingVideo = TrainingVideo(numFrames)
	w = 0
	h = 0

	for f in frames:
		# parse XML file... 
		detections, wCur, hCur = ProcessDetections(videoPath + '/' + f)
		w = wCur
		h = hCur
		frameNum = int(f.split('.')[0])
		for d in detections: trainingVideo.AddDetection(d, d.ID,frameNum)


	return trainingVideo, w, h




class ObjectTrack(object):
	def __init__(self, w, h):
		self.bboxes = []
		self.w = w
		self.h = h

	def Add(self, bbox):
		self.bboxes.append(np.array([bbox.x, bbox.y, bbox.w, bbox.h], dtype ='float64'))

	def Initialize(self, bbox):
		# Need to know the
		dx = bbox[0] if (bbox[0] < abs(self.w - bbox[0])) else (self.w - bbox[0])
		dy = bbox[1] if (bbox[1] < abs(self.w - bbox[1])) else (self.w - bbox[1])
		dw = 0
		dh = 0
		return np.array([dx, dy, dw, dh], dtype='float64')


	def Finalize(self):
		# Get DIFF; 
		bboxDiff = self.bboxes[:]
		i = 1

		#initialize FIRST bbox

		while i < len(bboxDiff):
			bboxDiff[i] = self.bboxes[i] - self.bboxes[i-1]
			i += 1

		return bboxDiff[1:]

	def size(self):
		return len(self.bboxes)
		

class DataProcessor(object):
	# collects VELOCITIES
	def __init__(self, w, h, cfg):
		self.w = w
		self.h = h
		self.cfg = cfg

	def Process(self, bboxList):
		tracks = []
		i = 0
		while i < len(bboxList):
			while i < len(bboxList) and bboxList[i] is None:
				i += 1

			tracks.append(ObjectTrack(self.w, self.h))

			while i < len(bboxList) and bboxList[i] is not None:
				tracks[-1].Add(bboxList[i])
				i += 1


		finalTracks = [t for t in tracks if t.size() >= 100+self.cfg.lookbackLength]

		return [oT.Finalize() for oT in finalTracks]



	def Augment(self, dataPoint):
		#datapoint should be in the form of list(np.array([dx, dy, dw, dh]))
		lookbackLength = self.cfg.lookbackLength
		current = np.array(dataPoint[lookbackLength:], dtype='float32')[0:(100-lookbackLength)]
		prev = np.array(dataPoint[(lookbackLength-1):-1], dtype='float32')[0:(100-lookbackLength)]
		past = np.array([dataPoint[i:(i+lookbackLength)] for i in range(len(dataPoint) - (lookbackLength))], dtype='float32')[0:(100-lookbackLength)]

		#so in the end we have a list of: [current, previous, past10]
		return [np.array(current, dtype='float32'), np.array(prev, dtype='float32'), np.array(past, dtype='float32')]



	def GetTrainingData(self, objectTracks):
		# Converts objectTracks data to numpy

		# objectracks is a DICT of tracks...
		tracks = []

		for tid, bboxes in objectTracks.items():
			#bboxes contains a LIST of bboxes 
			trackArrays = self.Process(bboxes)
			#if tracks are too short... don't process. 
			tracks += trackArrays



		return [self.Augment(t) for t in tracks]





def MakeBatch(trainingData):
	#print(trainingData[0][0])
	current = np.asarray([dat[0] for dat in trainingData], dtype='float32')
	prev = np.asarray([dat[1] for dat in trainingData], dtype='float32')
	past = np.asarray([dat[2] for dat in trainingData], dtype='float32')

	return [current, prev, past]