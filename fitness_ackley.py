import numpy as np

def fitness_ackley(genotype = [0.0]):
    """
    fitness_ackley:
        Calculates the fitness of a genotype concerning Ackley function
    
    Keyword arguments
    _________________
    
        genotype (double vector) :
            vector of genes (default: [0.0])

    Return
    ______

        the fitness of the genotype (double)
        
    """
    c1, c2 ,c3 = 20, 0.2, np.pi*2
    numGenes = np.float32(len(genotype))
    
    sum1 = np.sum( [ i ** 2 for i in genotype] )
    sum2 = np.sum( np.cos( c3 * i ) for i in genotype )
    
    term1 = (-c1) * np.exp( -c2 * ((1 / numGenes) * (sum1 ** ( 0.5 ))))
    term2 = -np.exp(( 1 / numGenes ) * sum2)

    err = term1 + term2 + c1 + np.exp(1)
    return (err)
if __name__ == '__main__':
	genotype = [0.0, 0.0, 0.0]
	print fitness_ackley( genotype )