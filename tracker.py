import numpy as np

from enum import Enum
from config import Cfg

import tensorflow as tf
import tf.keras as K


class TrackerState(Enum):
	NEW = 0
	INIT = 1
	ACTIVE = 2


class Tracker(object):
	def __init__(self, model_):
		self.cfg = Cfg()
		# Model - could be pre-trained... pre-sharing weights??? 

		self.model = model_
		# Most Recent Track
		self.memory = np.zeros(shape=(4, self.cfg.lookbackLength))
		self.bboxHistory = []
		self.state = TrackerState.INIT
		# Most Recent velocity...
		# Prediction 



	'''def GetIoU(self, x1, x2):
		#Get IoU with most recent frame... '''
		


	def GetScore(self, x):
		#x is a set of bbox coordinates
		if self.state is TrackerState.INIT: 
			# let score be the IoU
			return 0.0
		
		# use the model
		dx = np.array(x) - self.memory[-1, :]

		return self.model.Evaluate(dx)


	def Update(self, x):
		if self.state is TrackerState.NEW:
			self.bboxHistory.append(x)
			self.state = TrackerState.INIT
		elif self.state is TrackerState.INIT:
			#update state
			# Compute bounding box displacement
			dx = x - bboxHistory[-1]
			self.memory = np.tile(dx, (self.cfg.lookbackLength, 1))
			bboxHistory.append(x)
			self.state = TrackerState.ACTIVE
		else:
			dx = x - bboxHistory[-1]
			self.memory = np.concatenate(self.memory[1:,:], dx, axis=0)
			bboxHistory.append(x)







