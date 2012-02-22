"""
A set of functions for measuring the similarity between two identically sized
arrays.
"""
def euclidean(array1,array2):
	"""
	Returns the euclidian distance between the nth row of each array.
	"""
	import numpy as np

	distances = list()
	for n in range(array1.shape[0]):
		row_pairs = zip(array1[n,...],array2[n,...])
		orth_ds = [(x-y)**2 for x,y in row_pairs]
		distances.append(sum(orth_ds))
		
	return distances

