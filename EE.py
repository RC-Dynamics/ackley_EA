import numpy as np
from fitness_ackley import fitness_ackley as fitness

class SimpleEE:
	def __init__(self):
		self.genotype = [0.01 for _ in range(0, 15)]
		self.mutationStep = 0.8;
		self.stdDev = 0.2;
		self.hitVector = [0 for _ in range(0,5)]
		self.lastFitness = self.getFitness()
		self.iter = 0

	def mutation(self):
		hit  = np.sum(self.hitVector)
		if hit > 1 :
			self.stdDev /= self.mutationStep
		elif hit < 1:
			self.stdDev *= self.mutationStep

		newGenotype = [ i + np.random.normal(0.0, self.stdDev) for i in self.genotype] 
		if self.__getFitness(newGenotype) < self.__getFitness(self.genotype):
			self.genotype = newGenotype
			self.hitVector[self.iter%5] = 1;
		else:
			self.hitVector[self.iter%5] = 0;
		self.iter += 1
	
	def __getFitness(self, genotype):
		return fitness(genotype)

	def getFitness(self):
		return fitness(self.genotype)

if __name__ == '__main__':
	Niter = 10000
	SEE = SimpleEE()
	
	fitList = []
	for i in range(0, Niter):
		fitList.append(SEE.getFitness())
		SEE.mutation()
		
	print ("Initial Fitness: {}".format(fitList[0]))
	print ("Fitness After {} Iterations: {}".format(Niter, np.min(fitList)))
	print ("Min possible fitness: {}".format(fitness()))