import numpy as np


class Cfg(object):
	#define configuration parameters here
	def __init__(self):
		#defaults
		self.batchsize = 1 #MUST BE 1 to have stateful 
		self.featureLength = 4
		self.lookbackLength = 10
		self.timesteps = 1

		self.baseProbability = 0.25