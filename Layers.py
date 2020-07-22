import tensorflow as tf
import tensorflow_probability as tfp


class NormDist(tf.keras.layers.Layer):
	# layer for computing normal distribution parameters (mean and covariance matrix)

	def __init__(self, timesteps_, featureLength_, lookbackLength_, batchSize_):
		# call super..
		super(NormDist, self).__init__()
		self.batchSize = batchSize_
		self.timesteps = timesteps_
		self.featureLength = featureLength_
		self.lookbackLength = lookbackLength_


	def call(self, inputs):
		normParams = inputs[0] #Tensor
		pastFeatures = inputs[1] #Tensor
		curFeatures = inputs[2]

		# use slice to get normParams? yeah
		meanWeights = normParams[:,:,0:self.lookbackLength]
		stddev = normParams[:,:,(self.lookbackLength):]

		# Normalize standard deviation 
		stddev = tf.math.exp(stddev)

		# Normalize mean weights
		meanWeights, _ = tf.linalg.normalize(tf.math.exp(meanWeights), ord=1, axis=2) 

		#Repeat meanWeights, along the number of features 


		
		meanWeights = tf.repeat(tf.expand_dims(meanWeights, 3), self.featureLength, axis=3)

		mean = tf.math.reduce_mean(tf.math.multiply(meanWeights, pastFeatures), axis=2)

		#return ANOTHER tensor. 
		dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=stddev)

		return dist.log_prob(curFeatures)



'''class NormDistParams(tf.keras.layers.Layer):
	# layer for computing normal distribution parameters (mean and covariance matrix)

	def __init__(self, timesteps_, featureLength_, lookbackLength_, batchSize_):
		# call super..
		super(NormDist, self).__init__()
		self.batchSize = batchSize_
		self.timesteps = timesteps_
		self.featureLength = featureLength_
		self.lookbackLength = lookbackLength_


	def call(self, inputs):
		normParams = inputs[0] #Tensor
		pastFeatures = inputs[1] #Tensor

		# use slice to get normParams? yeah
		meanWeights = normParams[:,:,0:self.lookbackLength]
		stddev = normParams[:,:,(self.lookbackLength):]


		#Get weighted sequence, 

		#Repeat meanWeights, along the number of features 
		meanWeights = tf.repeat(tf.reshape(meanWeights, [1, self.timesteps, self.lookbackLength, 1]), self.featureLength, axis=3)

		mean = tf.math.reduce_mean(tf.math.multiply(meanWeights, pastFeatures), axis=2)


		return [mean, stddev]'''



