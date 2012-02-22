""" Simulate reward spaces. """
import numpy as np
import scipy.stats as stats

def random(N,p):
	"""
	Yield a single trials worth of random behavioral accuracy. P(1) = p.
	"""

	reward_space = np.random.binomial(1,p,N)
	for r in reward_space:
		yield r

	# state_space = np.array(state_space)
	# 
	# k = state_space.ndim
	# if k > 2:
	# 	raise ValueError('State spaces cannot exceed 2d.')
	# 
	# reward_space = np.zeros_like(state_space)
	# for col in range(k):
	# 	if k == 1:
	# 		state_mask = state_space != 0
	# 		reward_space[state_mask] = np.random.binomial(
	# 				1,p,sum(state_mask))
	# 	else:
	# 		state_mask = state_space[...,col] != 0
	# 		reward_space[state_mask,col] = np.random.binomial(
	# 				1,p,sum(state_mask))
	# 
	# return reward_space


def learn_logistic(N):
	"""
	Use state_space to create a reward array where p(1) increases by
	a (noisy) sampling of the logistic function simulating behavioral
	learning.
	"""

	
	# Simulate learning:
	# Create a noisy range for the CDF,
	trials = np.arange(.01,10,10/float(N))
	
	noise = stats.norm.rvs(size=N,scale=0.3)
	p_tmp = stats.norm.cdf(trials,3) + noise
	p_tmp[p_tmp < 0.] = 0
	p_tmp[p_tmp > 1.] = 1
		## p(1)  must be between 0..1
	p_values = p_tmp
		
	for p in p_values:
		yield int(np.random.binomial(1,p,(1)))
	



