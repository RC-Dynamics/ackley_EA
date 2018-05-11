import numpy as np
from random import random
from tqdm import tqdm
from fitness_ackley import fitness_ackley as fitness
import copy


class SimpleEA():

	def __init__(self, genotype, convergenceLimit = 10000):
		self.genotype = np.array(genotype, dtype=np.float64)
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
		elif hit < 1 and self.stdDev[0] > self.stdMinValue:
			self.stdDev[0] *= self.mutationStep

		son = self.genotype + np.random.normal(0.0, self.stdDev[0])
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
		self.genotype = np.array(genotype, dtype=np.float)
		self.mutationStep = 0.8
		self.stdDev = np.ones(30)*5
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
				self.stdDev[i] /= self.mutationStep
			elif hit < 1 and self.stdDev[i] >= self.stdMinValue:
				self.stdDev[i] *= self.mutationStep
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


class EA3:
	"""
    EA3 Dif:
		Bigger Population  
		Crossover 
		Auto-adaptive
    """
	def __init__(self, population, convergenceLimit = 1000):
		self.population = population
		self.popSize = len(population)
		self.numGenes = len(population[0])
		self.popFit = np.zeros((self.popSize))
		self.stdDev = (np.random.random((self.popSize, self.numGenes))*4) + 1

		self.iter = 0
		self.convergenceLimit = convergenceLimit
		self.convergenceCount = 0
		self.parent_index = []
		self.__init_fitness()
	
	def __init_fitness(self):
		for i, genotype in enumerate(self.population):
			self.popFit[i] = self.__getFitness(genotype)

	def forward(self):
		self.parent_selection()
		self.cross_over()
		self.mutation()
		self.childrens_selection()
		self.iter += 1
		return (np.min(self.popFit)),self.stopCondition()

	def parent_selection(self):
		"""
		Roulette Function
		"""
		self.parent_index = []
		fitSum = np.sum( 1/(self.popFit+1e-20) )
		
		for i in range(self.popSize*2):
			summ = 0
			choice = random() * fitSum
			for i in range(self.popSize):
				summ += 1/(self.popFit[i]+1e-20)
				if(choice <= summ):
					self.parent_index.append(i)
					break


	def cross_over(self):

		childrens = []
		childrensDev = []
		for i in range(self.popSize):
			idx, idy = self.parent_index[i * 2], self.parent_index[i * 2 + 1]
			child = (self.population[idx] + self.population[idy]) / 2.0
			childDev = (self.stdDev[idx] + self.stdDev[idy]) / 2.0  
			childrens.append(child)
			childrensDev.append(childDev)

		self.childrens = np.array(childrens)
		self.childrensDev = np.array(childrensDev)
		

	def mutation(self):
		self.childrens = self.childrens + np.random.normal(0.0, self.childrensDev)	
		childrensFit = np.zeros(self.popSize)
		for i, genotype in enumerate(self.childrens):
			childrensFit[i] = self.__getFitness(genotype)
		self.childrensFit = np.array(childrensFit)
	
	def childrens_selection(self):
		newPop = np.vstack((self.population, self.childrens))
		newFit = np.hstack((self.popFit, self.childrensFit))
		newDev = np.vstack((self.stdDev, self.childrensDev))

		fitOrd = newFit.argsort()
		newPop = newPop[fitOrd[::-1]]
		newDev = newDev[fitOrd[::-1]]
		newFit = newFit[fitOrd[::-1]]

		self.population = newPop[-self.popSize:]
		self.stdDev = newDev[-self.popSize:]
		self.popFit = newFit[-self.popSize:]
			
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
	num_iterations = 200000

	genotype = (np.random.random((1, 30))*30) - 15
	EA = EA3(genotype)
	tqdmBar = tqdm(range(num_iterations))
	for i in tqdmBar:
		minFit, end = EA.forward()
		tqdmBar.set_description("Fit: {}".format(minFit))

	# genotype1 = np.array((np.random.random((1, 30))*30) - 15)[0]
	# #EA = SimpleEA(genotype1)
	# EA = EA2(genotype1)
	# fitList = []
	# tqdmBar = tqdm( range(0, num_iterations))
	# for i in tqdmBar:
	# 	tqdmBar.set_description("Fit: {}, StdDev[0]: {}".format(EA.getFitness(), EA.stdDev[0]))
	# 	EA.mutation()
	# 	fitList.append(EA.getFitness())
	# 	if(EA.stopCondition()):
	# 		tqdmBar.close()
	# 		break
	
	# 	#print  EA.genotype, EA.getFitness()

	# print ("\n\nInitial Fitness: {}".format(fitList[0]))
	# print ("Fitness After {} Iterations: {}".format(num_iterations, np.min(fitList)))
