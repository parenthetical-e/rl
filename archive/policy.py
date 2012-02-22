""" Several action/state selection policies."""
import numpy as np
import rl

def decide(p_values,states):
	"""
	Several policies return only p_values.  This converts those to a
	decision.  For each entry in p_values either the matching paired-by
	-order state is returned or one of the other listed states is, which 
	depends, naturally, on p.
	"""

	choices = [0] * len(p_values) # init w/ 0
	state_names = list(set(states))
	
	# open Q: what is the p
	for ii,p in enumerate(p_values):
		if p == 1:
			choices[ii] = states[ii]
		elif np.rand.random() < p:
			choices[ii] = states[ii]
		else:
			pass
			# TODO and so when not p=1 and was not chosen
			# do what?

	return choices


def softmax(values,states,beta):
	"""
	Uses the softmax distribution to convert values into probabilities 
	of choosing each state in states, returning p_values, a list.
	"""
	from copy import deepcopy

	## Dealing with the (likely) ZeroDivisionError,
	## exp(0) = 1,
	## search Vs for 0 replacing w/ 0.000001
	if 0 in values:
		values_no0 = []
		for v in values:
			if v == 0:
				values_no0.append(0.000001)
			else:
				values_no0.append(v)
		else:
			values = deepcopy(values_no0)

	## Drop null conditions
	values = rl.misc.drop_null(values,states)
	states = rl.misc.drop_null(states,states)

	p_values = []
	names = list(set(states))
	if len(names) == 1:
		## If there is only one state
		## the calculation greatly 
		## simplfies.  Use that.
		softmax_1s = lambda x: np.exp(beta*x) / (1 + np.exp(beta*x))
		[p_values.append(softmax_1s(v)) for v in values]
	else:
		v_options = {}
		for n in names: v_options[n] = 0.000001
		for ii,v in enumerate(values):
			## Find v_options for this iteration.
			values_to_ii = values[:ii]
			values_to_ii.reverse()
			states_to_ii = states[:ii]
			states_to_ii.reverse()
			for n in set(states_to_ii):
				v_options[n] = values_to_ii[states_to_ii.index(n)]

#			print('***************************')
#			print('v_options: {0}',format(v_options))			
#			print('v',v)
#			print('s',states[ii])
#			print('v_op',v_options.values())

			## Softmax:
			p_0 = np.exp(beta*v)/sum(
					[np.exp(beta*op) for op in v_options.values()] )
			p_values.append(p_0)
						
	return p_values


def e_greedy(Vs,states,epsilon):
	"""
	UNTESTED

	An implementation of the e-greedy policy as described in Sutton and 
	Barto (1998). Epsilon must be between 0-1.  Returns a list of choices 
	(i.e. states), one for each states entry.
	"""
	
	import random as rand
	
	# Find the best option at current s
	# be greedy (use it) or pick randomly from the 
	# other choices
	# TODO Use defaultdict instead to handle new states?
	available_Vs = {}
	for s in set(states):
		available_Vs[s] = 0
	
	# find best v, store label and value
	best_v_s =  None# todo (v,s)
	choices = []
	for v,s in Vs,states:
		greed = True
		if rand.random() <= epsilon:
			greed = False

		if greed:
			pass
			# TODO y: is current v > best_v, return the choice		
		else:
			pass
			# n: pick one of the other v,s pairs and return that
	# for consistency instead return p_vals - 1 for choice e for epsilon	
	return choices


