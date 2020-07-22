import numpy as np

import tensorflow as tf

from Layers import NormDist
from config import Cfg



# ONE way to implement this 
class RAN(tf.keras.Model):
	#Define your keras model here... 

	def __init__(self, configs):
		# Pass in configuration parameters 
		super(RAN, self).__init__()

		# begin model definition
		self.conf = configs

		self.gruLayer = tf.keras.layers.GRU(self.conf.lookbackLength + self.conf.featureLength, return_sequences=True, stateful=True)

		self.denseLayer = tf.keras.layers.Dense(self.conf.lookbackLength + self.conf.featureLength)

		self.normDist = NormDist(self.conf.timesteps, self.conf.featureLength, self.conf.lookbackLength, self.conf.batchsize)





	def call(self, inputs):
		#allow multiple inputs 
		curFeatures = tf.convert_to_tensor(inputs[0])
		lastFeatures = tf.convert_to_tensor(inputs[1])
		pastFeatures = tf.convert_to_tensor(inputs[2])

		gruOut = self.gruLayer(lastFeatures)
		normParams = self.denseLayer(gruOut)

		return self.normDist([normParams, pastFeatures, curFeatures])


	



class RAN_wrapper(object):
	def __init__(self):
		# what is an RAN? 
		# External memory
		config = Cfg()
	
		self.conf = config
		#build model
		self.InitModel()
		self.InitTrainingModel()

	


	def InitModel(self):
		# Connect the layers 
		inputShape1 = (self.conf.timesteps, self.conf.featureLength)
		input1 = tf.keras.Input(shape=inputShape1, batch_size=1)

		inputShape2 = (self.conf.timesteps, self.conf.featureLength)
		input2 = tf.keras.Input(shape=inputShape2, batch_size=1)

		inputShape3 = (self.conf.timesteps, self.conf.lookbackLength, self.conf.featureLength)
		input3 = tf.keras.Input(shape=inputShape3, batch_size=1)

		self.gruLayer = tf.keras.layers.GRU(self.conf.lookbackLength + self.conf.featureLength, return_sequences=True, stateful=True)(input2)

		self.denseLayer = tf.keras.layers.Dense(self.conf.lookbackLength + self.conf.featureLength)(self.gruLayer)

		self.normDist = NormDist(self.conf.timesteps, self.conf.featureLength, self.conf.lookbackLength, self.conf.batchsize)([self.denseLayer, input3, input1])


		self.model = tf.keras.Model(inputs=[input1, input2, input3], outputs=self.normDist)



		opt = tf.keras.optimizers.Adam()
		self.model.compile(optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError) # with everything that we need 






	def InitTrainingModel(self):

		inputShape1 = (None, self.conf.featureLength)
		input1 = tf.keras.Input(shape=inputShape1)

		inputShape2 = (None, self.conf.featureLength)
		input2 = tf.keras.Input(shape=inputShape2)

		inputShape3 = (None, self.conf.lookbackLength, self.conf.featureLength)
		input3 = tf.keras.Input(shape=inputShape3)

		self.gruLayer = tf.keras.layers.GRU(self.conf.lookbackLength + self.conf.featureLength, return_sequences=True)(input2)

		self.denseLayer = tf.keras.layers.Dense(self.conf.lookbackLength + self.conf.featureLength)(self.gruLayer)

		self.normDist = NormDist(self.conf.timesteps, self.conf.featureLength, self.conf.lookbackLength, self.conf.batchsize)([self.denseLayer, input3, input1])


		self.trainingModel = tf.keras.Model(inputs=[input1, input2, input3], outputs=self.normDist)



		opt = tf.keras.optimizers.Adam()
		self.trainingModel.compile(optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError()) # with everything that we need 

	


	def LoadWeights(self, weights):
		#LOAD WEIGHTS 
		self.model.load_weights(weights)


	def LoadTrainingWeights(self, weights):
		#LOAD WEIGHTS 
		self.trainingModel.load_weights(weights)


	def SaveWeights(self, savePath):
		#LOAD WEIGHTS 
		self.model.save_weights(path=savePath, save_format="h5")


	def SaveTrainingWeights(self, savePath):
		#LOAD WEIGHTS 
		self.trainingModel.save_weights(filepath=savePath, save_format="h5")



	def Summary(self):
		self.model.summary()


	def Train(self, data, output):
		self.trainingModel.fit(x=data, y=output, batch_size=64, epochs=100)



	def Evaluate(self, dx, dx_prev, dx_mem):
		#with cfg as prediction parameters, perform prediction on "x"
		#i.e. compute P(x[t] | x[1:t-1]) -> joint probability calculation 
		dx = np.array([[dx]])
		dx_prev = np.array([[dx_prev]])
		dx_mem = np.array([[dx_mem]])

		res = self.model.predict([dx, dx_prev, dx_mem])

		return res[0][0]


