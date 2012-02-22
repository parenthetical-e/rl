"""
Prediction error (loss) functions for reinforcement learning.  
"""

def delta(r,v):
	return r - v
	
def delta_dist(r,v,D):
	return r/D - v
