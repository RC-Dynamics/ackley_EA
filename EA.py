import numpy as np
import random
from tqdm import tqdm
from fitness_ackley import fitness_ackley as fitness
import copy


class SimpleEA():

	def __init__(self, genotype, convergenceLimit = 10000):
		self.genotype = genotype
		self.mutationStep = 0.8
		self.stdDev = [5]
		self.hitVector = [0 for _ in range(0,5)]
		self.bestFitness = self.__getFitness(genotype)
		self.lastFitness = self.bestFitness
		self.iter = 0
		self.stdMinValue = 1

		self.convergenceLimit = convergenceLimit
		self.convergenceCount = 0
		
	def mutation(self):
		hit  = np.sum(self.hitVector)
		if hit > 1:
			self.stdDev[0] /= self.mutationStep
		elif hit < 1:
			self.stdDev[0] *= self.mutationStep

		son = [ i + np.random.normal(0.0, self.stdDev[0]) for i in self.genotype]
		sonFitness = self.__getFitness(son)
		if  sonFitness <= self.bestFitness:			
			self.genotype = son
			self.bestFitness = sonFitness
			self.hitVector[self.iter%5] = 1

		else:
			self.hitVector[self.iter%5] = 0
		if(self.lastFitness == self.bestFitness):
				self.convergenceCount += 1
		else:
			self.convergenceCount = 0
		self.lastFitness = self.bestFitness

		self.iter += 1
	
	def __getFitness(self, genotype):
		return fitness(genotype)

	def getFitness(self):
		return fitness(self.genotype)

	def stopCondition(self):
		if(self.convergenceCount >= self.convergenceLimit):
			return True
		else:
			return False


class EA2:
	def __init__(self, genotype, convergenceLimit = 1000):
		self.genotype = genotype
		self.mutationStep = [0.8 for _ in range(0,30)]
		self.stdDev = [5 for _ in range(0, 30)]
		self.hitVector = [[0, 0, 0, 0, 0] for _ in range(0,30)]
		
		self.bestFitness = self.__getFitness(genotype)
		self.lastFitness = self.bestFitness
		self.iter = 0
		self.stdMinValue = 1

		self.convergenceLimit = convergenceLimit
		self.convergenceCount = 0

	def mutation(self):
		if self.iter % 250 == 250-1:
			self.stdMinValue /= 5

		for i in range(30):
			hit  = np.sum(self.hitVector[i])
			if hit > 1:
				self.stdDev[i] /= self.mutationStep[i]
			elif hit < 1 and self.stdDev[i] >= self.stdMinValue:
				self.stdDev[i] *= self.mutationStep[i]
			son = copy.deepcopy(self.genotype)
			son[i] += np.random.normal(0.0, self.stdDev[i])
			sonFitness = self.__getFitness(son)

			if  sonFitness <= self.bestFitness:			
				self.genotype = son
				self.bestFitness = sonFitness
				self.hitVector[i][self.iter%5] = 1
			else:
				self.hitVector[i][self.iter%5] = 0
			
		if(self.lastFitness == self.bestFitness):
				self.convergenceCount += 1
		else:
			self.convergenceCount = 0
		self.lastFitness = self.bestFitness
		self.iter += 1
	
	def __getFitness(self, genotype):
		return fitness(genotype)

	def getFitness(self):
		return self.bestFitness

	def stopCondition(self):
		if(self.convergenceCount >= self.convergenceLimit):
			return True
		else:
			return False



if __name__ == '__main__':
	num_iterations = 20000

	genotype = [(random.random()*30 ) - 15  for _ in range(0, 30)]
	
	EA = EA(genotype)
	
	fitList = []
	tqdmBar = tqdm( range(0, num_iterations))
	for i in tqdmBar:
		tqdmBar.set_description("Fit: {}, StdDev[0]: {}".format(EA.getFitness(), EA.stdDev[0]))
		EA.mutation()
		fitList.append(EA.getFitness())
		if(EA.stopCondition()):
			tqdmBar.close()
			break
	
		#print  EA.genotype, EA.getFitness()

	print ("\n\nInitial Fitness: {}".format(fitList[0]))
	print ("Fitness After {} Iterations: {}".format(num_iterations, np.min(fitList)))
	print ("Final Solution:")

