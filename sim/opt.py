def ml_score(p_values):
	"""
	Calculate the log-likelihood score
	"""
	from copy import deepcopy
	
	log_L = sum([np.log(p) for p in p_values])
	return log_L
