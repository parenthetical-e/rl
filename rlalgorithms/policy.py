"""
Policies for selecting the next state
"""
import numpy as np
import bisect as bi

# TODO fix so this uses use only attr and params
# TODO vectorize
def softmax(self,beta):
	""" 
	Use the current and past v and the logistic function 
	to guess the state with the highest probability of winning.
	"""
	from collections import deepcopy
	
	# Find all possible states
	known_states = list(set(history['s']))
	known_states.sort()
	
	# Get the values for those states
	latest_v = np.zeros(len(known_states))
	for i,ks in enumerate(known_states):
		latest_v[i] = self.get_from_history(self,ks,'v',1)
	
	# Get p for each latest_v (vectorized)
	ps = np.exp(latest_v*beta) / np.sum(np.exp(beta*latest_v))
	
	# TODO p weighted sampling of known_states goes here
	# http://prxq.wordpress.com/2006/04/17/the-alias-method/
	# for an example of how to do this.  None of pythons libs seem
	# to support it.
		
	# Return the state match p_use
	known_states = np.asarray(known_states)	
	return known_states[p_used == ps], p_u


def e_greedy(Vs,Ss,epsilon):
	s = None
	max_V = max(Vs)
	if np.random.rand() > epsilon:
		loc = Vs.index(max_V)
		s = Ss[loc]
	else:
		Vs_cp = Vs.copy().remove(max_V)
		
	pass	