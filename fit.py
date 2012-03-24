"""Fit parameters to behavoiral data"""
import numpy as np
import rl

def ml_delta(acc,states,res):
	"""
	Use maximum likelihood to find the best alpha and beta values for the 
	delta learning rule.
	"""
	from itertools import product
	from copy import deepcopy

	params = ((0.01,1),(0.01,5))
		# alpha, beta ranges

	param_values = [np.arange(*par,step=res) for par in params]
	all_unique_param = product(*param_values)
		# http://www.technovelty.org/code/python/asterisk.html
		# for a nice tutorial on asterices and python
	
	best_log_L = None
	best_par = ()
#	p_log = {}  
		# a log of all p_values
	p_cnt = 0
	for alpha,beta in all_unique_param:
		v_dict, rpe_dict = rl.reinforce.b_delta(acc,states,alpha)
		v = rl.misc.unpack(v_dict,states)

		p_values = rl.policy.softmax(v,states,beta)
		log_L = sum([np.log(p) for p in p_values])
#		p_log[p_cnt] = p_values
		
		if p_cnt == 0: 
			best_log_L = deepcopy(log_L)
			best_par = (alpha,beta)
				# init best_log_L and best_par
				# on first go


		if log_L > best_log_L:
			print('Improvement - L:{0}, alpha:{1}, beta:{2}'.format(
				log_L,alpha,beta))
			best_log_L = deepcopy(log_L)
			best_par = (alpha,beta)

		p_cnt += 1

	return (best_par,best_log_L)
	










































