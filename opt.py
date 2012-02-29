def ml_score(p_values):
	"""
	Calculate the log-likelihood score
	"""
	from copy import deepcopy
	
	# Vectorize
	return np.sum(np.log(np.array(p_values)))
