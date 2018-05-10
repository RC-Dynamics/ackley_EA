import numpy as np
import random
from tqdm import tqdm
from fitness_ackley import fitness_ackley as fitness
import copy


class SimpleEE():

	def __init__(self, genotype):
		self.genotype = genotype
		self.mutationStep = 0.8
		self.stdDev = 5
		self.hitVector = [0 for _ in range(0,5)]
		self.lastFitness = self.getFitness()
		self.iter = 0
		self.stdMinValue = 1
		
	def mutation(self):
		hit  = np.sum(self.hitVector)
		if hit > 1:
			self.stdDev /= self.mutationStep
		elif hit < 1:
			self.stdDev *= self.mutationStep

		newGenotype = [ i + np.random.normal(0.0, self.stdDev) for i in self.genotype] 
		if self.__getFitness(newGenotype) <= self.__getFitness(self.genotype):			
			self.genotype = newGenotype
			self.hitVector[self.iter%5] = 1

		else:
			self.hitVector[self.iter%5] = 0

		self.iter += 1
	
	def __getFitness(self, genotype):
		return fitness(genotype)

	def getFitness(self):
		return fitness(self.genotype)

class EE2:
	def __init__(self, genotype):
		self.genotype = genotype
		self.mutationStep = [0.8 for _ in range(0,30)];
		self.stdDev = [5 for _ in range(0, 30)];
		self.hitVector = [[0, 0, 0, 0, 0] for _ in range(0,30)]
		
		self.lastFitness = self.getFitness()
		self.iter = 0
		self.stdMinValue = 1

	def mutation(self):
		if self.iter % 250 == 250-1:
			self.stdMinValue /= 5

		for i in range(30):
			hit  = np.sum(self.hitVector[i])
			if hit > 1:
				self.stdDev[i] /= self.mutationStep[i]
			elif hit < 1 and self.stdDev[i] >= self.stdMinValue:
				self.stdDev[i] *= self.mutationStep[i]
			newGenotype = copy.deepcopy(self.genotype)
			newGenotype[i] += np.random.normal(0.0, self.stdDev[i])

			if self.__getFitness(newGenotype) <= self.__getFitness(self.genotype):			
				self.genotype = newGenotype
				self.hitVector[i][self.iter%5] = 1

			else:
				self.hitVector[i][self.iter%5] = 0

		self.iter += 1
	
	def __getFitness(self, genotype):
		return fitness(genotype)

	def getFitness(self):
		return fitness(self.genotype)



if __name__ == '__main__':
	num_iterations = 20000
	#genotype = [0.1  for _ in range(0, 30)]
	genotype = [(random.random()*30 ) - 15  for _ in range(0, 30)]
	
	SEE = EE2(genotype)
	
	fitList = []
	tqdmBar = tqdm( range(0, num_iterations) )
	for i in tqdmBar:
		fitList.append(SEE.getFitness())
		#tqdmBar.set_description("Fit: {:0.2f}, StdDev: {:0.2f}".format(SEE.getFitness(), SEE.stdDev))
		tqdmBar.set_description("Fit: {}, StdDev: {}".format(SEE.getFitness(), SEE.stdDev[0]))
		
		SEE.mutation()

		#print  SEE.genotype, SEE.getFitness()

	print ("\n\nInitial Fitness: {}".format(fitList[0]))
	print ("Fitness After {} Iterations: {}".format(num_iterations, np.min(fitList)))
	print ("Final Solution:")

