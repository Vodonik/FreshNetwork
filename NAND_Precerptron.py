import math as m
import numpy as np

dataSet_NAND = [
	[[0, 0], 1],
	[[0, 1], 1],
	[[1, 0], 1],
	[[1, 1], 0]
]

dataSet_AND = [
	[[0, 0], 0],
	[[0, 1], 0],
	[[1, 0], 0],
	[[1, 1], 1]
]

dataSet_XOR = [
	[[0, 0], 0],
	[[0, 1], 1],
	[[1, 0], 1],
	[[1, 1], 0]
]

class Perceptron():
	def __init__(self, inputSize):
		self.bias = np.random.randn(1)
		self.weights = np.random.randn(inputSize)
		self.output = 0
		
	def Activate(self, inputs):
		self.inputs = inputs
		wSum = np.dot(self.weights, self.inputs)
		self.output = self.SigmoidF(wSum)
			
	def PerceptronF(self, sum):
		if sum + self.bias[0] <= 0:
			return 0
		else:
			return 1
			
	def SigmoidF(self, sum):
		return 1.0 / (1.0 + m.exp(-sum - self.bias))		

class Network():
	def __init__(self, config):
		self.numberOfLayers = len(config)
		self.layersSizes = config
		self.Initialize()
		self.eta = 0.05
		
	def Initialize(self):
		#empty "zeroeth" layer to enable generating input size for first network layer
		self.layers = [[]]
		for layer in self.layersSizes:
			layerNeurons = []
			for i in range(0, layer):
				layerNeurons.append(Perceptron(len(self.layers[-1])))
			self.layers.append(layerNeurons)
			
	def Forward(self, dataSet):
		for i in range(0, len(self.layers[1])):
			self.layers[1][i].output = dataSet[i]
			
		previousLayerOutput = dataSet
		#i starts with '2' in order to skip zeroeth and first(input) layer
		for i in range(2, len(self.layers)):
			currentLayerOutput = []
			for neuron in self.layers[i]:
				neuron.Activate(previousLayerOutput)
				currentLayerOutput.append(neuron.output)
			previousLayerOutput = currentLayerOutput
		
	def Backward(self, grad):
		for i in range(1, len(self.layers)):
			for j in range(0, len(self.layers[-i])):
				neuronOfInterest = self.layers[-i][j]
				neuronOutput = neuronOfInterest.output
				err = grad - neuronOutput
				
				for k in range(0, len(neuronOfInterest.weights)):
					neuronOfInterest.weights[k] += self.eta * err * neuronOfInterest.inputs[k]
					
				neuronOfInterest.bias[0] += self.eta * err
		
	def Train(self, iterations, dataSet):
		for i in range(0, iterations):
			ds = dataSet[np.random.randint(0, 4)]
			self.Forward(ds[0])
			self.Backward(ds[1])
				
def ListNetwork():
	for layer in net.layers:
		for neuron in layer:
			print(len(neuron.weights))

def Test(n, dataSet):
	iter = 10
	guessed = 0
	for i in range(0, iter):
		ds = dataSet[np.random.randint(0, 4)]
		n.Forward(ds[0])
		result = n.layers[-1][0].output
		actual = ds[1]
		print(actual, result)
		guessed += 1 if abs(actual - result) < 0.02 else 0
		
	print("Guessed: " + str(guessed) + "/" + str(iter))
			
net = Network([2, 2, 1])

activeDataSet = dataSet_XOR

net.Train(100000, activeDataSet)
Test(net, activeDataSet)
