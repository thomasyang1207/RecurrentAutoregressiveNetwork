import tensorflow as tf
import numpy as np
import random

from RAN import RAN_wrapper

from DataProcessing import *


#Initialization 
model = RAN_wrapper()
model.InitTrainingModel()


trainPath = '/home/thomas/DeepLearning/RAN/train/'
modelSavePath = '/home/thomas/DeepLearning/RAN/Models/trainWeights.h5'

model.LoadTrainingWeights(modelSavePath)


tf.get_logger().setLevel('ERROR')


#Training Loop 
trainingSets = GetTrainingSets(trainPath)


for epoch in range(100):
	for tSet in trainingSets:
		# for each training set
		setPath = trainPath + '/' + tSet
		trainingVideos = GetVideos(setPath)
		random.shuffle(trainingVideos)
		trainingData = []
		for vid in trainingVideos:
			processedVid, w, h = ProcessVideo(setPath + '/' + vid)
			dataProcessor = DataProcessor(w, h, Cfg())
			trainingData += dataProcessor.GetTrainingData(processedVid.TrackedObjects)


		# Convert to Batch form...
		lengths = set()
		for dat in trainingData:
			lengths.add(dat[0].shape)


		#print(lengths)

		#print(trainingData[0])
		trainingBatch = MakeBatch(trainingData)


		out = np.zeros(shape=(np.shape(trainingBatch[0])[0], np.shape(trainingBatch[0])[1]), dtype='float32')

		model.Train(data=trainingBatch, output=out)
		
		model.SaveTrainingWeights(modelSavePath)





