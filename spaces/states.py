""" Generate state spaces """
import numpy as np

def random(N,k):
	"""
	Randomly return a state between 1 and k.  N is the total number of states.
	"""
	if N%k != 0:
		raise ValueError('k must evenly divide N.')
	
	states = range(1,k+1) * int(N/k)
	np.random.shuffle(states)
	for s in states:
		yield s
