from DataProcessing import *
from config import Cfg


videoPath = '/home/thomas/DeepLearning/RAN/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00011000'
vid, w, h = ProcessVideo(videoPath)

dataProcessor = DataProcessor(w, h, Cfg())
trainingData = dataProcessor.GetTrainingData(vid.TrackedObjects)


trainingBatch = MakeBatch(trainingData)

print(trainingBatch)