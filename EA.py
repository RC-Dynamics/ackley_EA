import numpy as np
import matplotlib.pyplot as plt
from random import random 
from tqdm import tqdm
from fitness_ackley import fitness_ackley as fitness
import copy


class SimpleEA():

	def __init__(self, convergenceLimit = 10000):
		print ("Simple Evolutionary Algorithm")
		genotype = [(random()*30) - 15  for _ in range(0, 30)]
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
		
	def forward(self): # Only Mutation
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
	def getNumOfConvergence(self):
		return self.convergenceCount
	def getConvergenceLimit(self):
		return self.convergenceLimit


class EA:
	def __init__(self, convergenceLimit = 1000):
		print ("Evolutionary Algorithm")
		genotype = [(random()*30) - 15  for _ in range(0, 30)]
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

	def forward(self): # Only Mutation
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
	def getNumOfConvergence(self):
		return self.convergenceCount

	def getConvergenceLimit(self):
		return self.convergenceLimit
	
	def getGenotype(self):
		return self.genotype


class RefinedEA:
	"""
    EA3 Dif:
		Bigger Population  
		Crossover 
		Auto-adaptive
    """
	def __init__(self, convergenceLimit = 1000):
		print ("Refined Evolutionary Algorithm")
		population = (np.random.random((100,30))*30) - 15
		self.population = population
		self.popSize = len(population)
		self.numGenes = len(population[0])
		self.tau = 1/(np.sqrt(self.numGenes))
		self.popFit = np.zeros((self.popSize))
		self._stdDev = (np.random.random((self.popSize, self.numGenes))*4) + 1
		self.stdDev = self._stdDev[0]
		self.iter = 0
		self.convergenceLimit = convergenceLimit
		self.convergenceCount = 0
		self.parent_index = []
		self.bestOldFitness = 0
		self.convergenceCount = 0
		self.convergenceLimit = convergenceLimit

		self.__init_fitness()
		self.bestFitness = np.min(self.popFit)
	
	def __init_fitness(self):
		for i, genotype in enumerate(self.population):
			self.popFit[i] = self.__getFitness(genotype)


	def forward(self):
		self.parent_selection()
		self.cross_over()
		self.mutation()
		self.childrens_selection()
		self.iter += 1
		self.stdDev = self._stdDev[0]
		return (np.min(self.popFit), np.max(self._stdDev)),self.stopCondition()

	def parent_selection(self):
		"""
		Roulette Function
		"""
		self.parent_index = []
		fitSum = np.sum( 1/(self.popFit+0.1) )
		
		for i in range(self.popSize*2):
			summ = 0
			choice = random() * fitSum
			for i in range(self.popSize):
				summ += 1/(self.popFit[i]+0.1)
				if(choice <= summ):
					self.parent_index.append(i)
					break


	def cross_over(self):

		childrens = []
		childrensDev = []
		for i in range(self.popSize):
			idx, idy = self.parent_index[i * 2], self.parent_index[i * 2 + 1]
			select =  np.random.randint(2, size=len(self.population[idx]) )
			nselect = 1 - select 
			selectDev =  np.random.randint(2, size=len(self.population[idx]) )
			nselectDev = 1 - selectDev 
			child = (self.population[idx] * select) + (self.population[idy] * nselect)
			childDev = (self._stdDev[idx] * selectDev) + ( self._stdDev[idy] * nselectDev )  
			#childDev = (self._stdDev[idx] + self._stdDev[idy] )/2.0  
			childrens.append(child)
			childrensDev.append(childDev)

		self.childrens = np.array(childrens)
		self.childrensDev = np.array(childrensDev)
		

	def mutation(self):
		self.childrensDev *= np.exp(self.tau * np.random.normal(0.0, 1))
		self.childrens = self.childrens + np.random.normal(0.0, self.childrensDev)	
		childrensFit = np.zeros(self.popSize)
		for i, genotype in enumerate(self.childrens):
			childrensFit[i] = self.__getFitness(genotype)
		self.childrensFit = np.array(childrensFit)
	
	def childrens_selection(self):
		self.bestOldFitness = np.min(self.popFit)
		newPop = np.vstack((self.population, self.childrens))
		newFit = np.hstack((self.popFit, self.childrensFit))
		newDev = np.vstack((self._stdDev, self.childrensDev))

		fitOrd = newFit.argsort()
		
		newPop = newPop[fitOrd[::-1]]
		newDev = newDev[fitOrd[::-1]]
		newFit = newFit[fitOrd[::-1]]
		
		self.population = newPop[-self.popSize:]
		self._stdDev = newDev[-self.popSize:]
		self.popFit = newFit[-self.popSize:]

		self.bestFitness = np.min(self.popFit)
			
	def __getFitness(self, genotype):
		return fitness(genotype)

	def getFitness(self):
		return self.bestFitness

	def stopCondition(self):
		if( self.bestFitness >= self.bestOldFitness):
			self.convergenceCount += 1
			if self.convergenceCount > self.convergenceLimit:
				return True
		else:
			self.convergenceCount = 0
		return False
		
	def getNumOfConvergence(self):
		return self.convergenceCount
	
	def getConvergenceLimit(self):
		return self.convergenceLimit


if __name__ == '__main__':
	num_iterations = 20000

	# EA = SimpleEA()
	# EA = EA()
	EA = RefinedEA()

	
	
	convergeFitness = []
	numOfConvergences = []
	minNumIteration = []
	num_tests = 30

	for j in range(num_tests):
		fitList = []
		tqdmBar = tqdm( range(0, num_iterations))
		for i in tqdmBar:
			tqdmBar.set_description("Fit: {}, StdDev[0]: {}".format(EA.getFitness(), EA.stdDev[0]))
			EA.forward()
			fitList.append(EA.getFitness())
			if(EA.stopCondition()):
				tqdmBar.close()
				break

		print ("Initial Fitness: {}".format(fitList[0]))
		print ("Fitness After {} iterations: {}\n\n".format(i, EA.getFitness()))
		# print ("Solution: {}\n".format(EA.getGenotype()))

		convergeFitness.append(EA.getFitness())
		minNumIteration.append(i-EA.getNumOfConvergence()+1)

		EA.__init__()
	
	convergeFitness_mean = np.mean(convergeFitness)
	minNumIteration_mean = np.mean(minNumIteration)

	convergeFitness_std = np.std(convergeFitness)
	minNumIteration_std = np.std(minNumIteration)

	print ("\nAlgorithm Converged in {} / {} -- Converge condition of {} iterarions".format(len([x for x in convergeFitness if x < 0.1e-10]), num_tests, EA.getConvergenceLimit()))
	print ("Mean Fitness: {}| Mean of minimun Iterarions: {}".format(convergeFitness_mean, minNumIteration_mean))
	print ("Std Fitness: {}| Std of minimun Iterarions: {}\n".format(convergeFitness_std, minNumIteration_std))

	fig, axs = plt.subplots(ncols=2)

	# basic plot
	axs[0].boxplot(convergeFitness)
	axs[0].set_title('Converge Fitness')

	# change outlier point symbols
	axs[1].boxplot(minNumIteration)
	axs[1].set_title('Min of Iterations to Converge')
	plt.show()
