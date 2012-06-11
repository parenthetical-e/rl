""" A set of vectorized functions for measuring similarity. """
import numpy as np

def category_prototype(x1, x2, u1, u2):
	""" Return the 2d similarity between a lists of examplars 
	(<x1> and <x2>) and their means (<u1> and <u2>)
	
	Similarity is measured by:
		s = exp( sqrt((x1 - u1)^2) + sqrt((x2 - u2)^2) ).
	
	I.e. the exponential of the euclidian distance, which Shepard 
	(Science, 1988) demonstrated acts as a 'universal law of
	generalization'.
	"""
	x1 = np.array(x1)
	x2 = np.array(x2)
	u1 = np.array(u1)
	u2 = np.array(u2)
	
	if (x1.shape != u1.shape) or (u1.shape[0] != 1):
		raise ValueError(
			"x1 and u1 must be the same shape or u1 must be a scalar")

	if (x2.shape != u2.shape) or (u2.shape[0] != 1):
		raise ValueError(
			"x2 and u2 must be the same shape or u2 must be a scalar")
	
	# TODO is this math right?  Double check that the subtraction below
	# does the right thing...
	return np.exp(l2(x1 - u1, x2 - u2))


def l2(x1, x2, axis=0):
    """
    Returns the 2d euclidian distance (L2) between x1 and x2.
    """

    x1 = np.array(x1)
    x2 = np.array(x2)
    distances = np.sqrt(x1 ** 2 + x2 ** 2)

    return distances
