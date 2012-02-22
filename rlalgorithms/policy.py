"""
Policies for selecting the next state
"""
import numpy as np
import bisect as bi

def softmax(s,Vs,Ss,beta):
	p_denom = sum([np.exp(beta*V_i) for V_i in Vs])
	p_Vs = [np.exp(beta*v)/p_denom for V_i in Vs]
	p_s = 0
	s_out = None
	if s is not None:
		# Do a waeighted sompling of Ss using p_Vs. 
		# 
		# Code modified from:
		# http://eli.thegreenplace.net/2010/01/22/weighted-random
		# -generation-in-python/ 
	    totals = []
	    running_total = 0
	    for w in p_Vs:
	        running_total += w
	        totals.append(running_total)
		rnd = np.random.rand() * running_total
		loc = bi.bisect_right(totals, rnd)
		s_out = Ss[loc]
		p_s = p_Vs[loc]
	else:
		# If s was defined (i.e. not None) just use it.
		s_out = s.copy()
		p_s = p_Vs[Ss.index(s)]
		
	return s_out, p_s


def e_greedy(Vs,Ss,epsilon):
	s = None
	max_V = max(Vs)
	if np.random.rand() > epsilon:
		loc = Vs.index(max_V)
		s = Ss[loc]
	else:
		Vs_cp = Vs.copy().remove(max_V)
		
	pass	