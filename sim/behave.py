from collections import defaultdict
from rl.base import Agent
	
class Behave(Agent):
	""" A subclass of Agent with mojo for fitting behavioral data. """
	
	def __init__(self):
		self.best_log_L = -0.000000000000000000000001
			## Used by .fit():
			## it can't be worse than this, practically anyway
	
	
	def fit(self,resolution):
		"""
		Optimize the value update step size (alpha) and choice probability
		(beta).
		"""
		from itertools import product
		from copy import deepcopy
		from rl.sim.opt import ml_score
				
		# Create a list of all possible *unique* alpha and beta value
		# within the specified resolution.
		params = ((0.01,1),(0.01,5))
		param_values = [np.arange(*par,step=resolution) for par in params]
		all_unique_param = product(*param_values)

		# Find the best (by ML) alpha and beta using all_unique_param
		best_ab = ()
		for alpha,beta in all_unique_param:
			self.history = defaultdict(list)
			self.value_params['alpha'] = alpha
			self.policy_params['beta'] = beta
			self.run()
			log_L = ml_score(self.history['p'])
			if log_L > self.best_log_L:
				self.best_log_L = deepcopy(log_L)
				best_ab = deepcopy(alpha,beta)
		
		# One final run using the optimal alpha and beta 
		# thus seting all other attr to their optimal values
		# too.
		self.value_params['alpha'] = best_ab[0]
		self.policy_params['beta'] = best_ab[1]
		self.run()
	