#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'an ANN module'
__author__='lrcno6'
import numpy
import scipy.special
class NeuralNetwork:
	def __init__(self,layers,activation_function=scipy.special.expit):
		self.activation_function=activation_function
		self.weight=[numpy.random.normal(0,pow(layers[i],-0.5),(layers[i],layers[i-1])) for i in range(1,len(layers))]
	def __calc(self,inputs):
		outputs=[inputs]
		for i in self.weight:
			inputs=self.activation_function(numpy.dot(i,inputs))
			outputs.append(inputs)
		return outputs
	def query(self,inputs):
		inputs=numpy.array(inputs,ndmin=2).T
		return self.__calc(inputs)[-1]
	def train(self,inputs,targets,learning_rate):
		inputs=numpy.array(inputs,ndmin=2).T
		targets=numpy.array(targets,ndmin=2).T
		outputs=self.__calc(inputs)
		errors=targets-outputs[-1]
		for i in range(len(outputs)-1,0,-1):
			self.weight[i-1]+=learning_rate*numpy.dot(errors*outputs[i]*(1-outputs[i]),numpy.transpose(outputs[i-1]))
			errors=numpy.dot(self.weight[i-1].T,errors)