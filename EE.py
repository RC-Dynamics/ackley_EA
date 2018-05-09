import numpy as np
import random
from tqdm import tqdm
from fitness_ackley import fitness_ackley as fitness
import copy

class SimpleEE:
	def __init__(self):
		self.genotype = [0.1  for _ in range(0, 30)]
		#self.genotype = [(random.random()*30 ) - 15  for _ in range(0, 30)]
		self.mutationStep = 0.8;
		self.stdDev = 5;
		self.hitVector = [0 for _ in range(0,5)]
		self.lastFitness = self.getFitness()
		self.iter = 0
		self.stdMinValue = 1e-2

	def mutation(self):
		hit  = np.sum(self.hitVector)
		if hit > 1:
			self.stdDev /= self.mutationStep
		elif hit < 1:
			self.stdDev *= self.mutationStep

		newGenotype = [ i + np.random.normal(0.0, self.stdDev) for i in self.genotype] 
		if self.__getFitness(newGenotype) <= self.__getFitness(self.genotype):			
			self.genotype = newGenotype
			self.hitVector[self.iter%5] = 1;

		else:
			self.hitVector[self.iter%5] = 0;

		self.iter += 1
	
	def __getFitness(self, genotype):
		return fitness(genotype)

	def getFitness(self):
		return fitness(self.genotype)

class EE:
	def __init__(self):
		#self.genotype = [0.1  for _ in range(0, 30)]
		self.genotype = [(random.random()*30 ) - 15  for _ in range(0, 30)]
		self.mutationStep = [0.8 for _ in range(0,30)];
		self.stdDev = [5 for _ in range(0, 30)];
		self.hitVector = [[0, 0, 0, 0, 0] for _ in range(0,30)]
		
		self.lastFitness = self.getFitness()
		self.iter = 0
		self.stdMinValue = 1e-2

	def mutation(self):

		for i in range(30):

			hit  = np.sum(self.hitVector[i])
			if hit > 1:
				self.stdDev[i] /= self.mutationStep[i]
			elif hit < 1:
				self.stdDev[i] *= self.mutationStep[i]
			newGenotype = copy.deepcopy(self.genotype)
			newGenotype[i] += np.random.normal(0.0, self.stdDev[i])

			if self.__getFitness(newGenotype) <= self.__getFitness(self.genotype):			
				self.genotype = newGenotype
				self.hitVector[i][self.iter%5] = 1;

			else:
				self.hitVector[i][self.iter%5] = 0;

		self.iter += 1
	
	def __getFitness(self, genotype):
		return fitness(genotype)

	def getFitness(self):
		return fitness(self.genotype)



if __name__ == '__main__':
	Niter = 20000

	SEE = SimpleEE()
	
	fitList = []
	for i in tqdm( range(0, Niter) ):
		fitList.append(SEE.getFitness())
		SEE.mutation()
		#print  SEE.genotype, SEE.getFitness()

	print ("\n\nInitial Fitness: {}".format(fitList[0]))
	print ("Fitness After {} Iterations: {}".format(Niter, np.min(fitList)))
	print ("Final Solution:")
	#print (SEE.genotype)


