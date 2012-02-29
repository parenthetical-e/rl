"""
Prediction error (loss) functions for reinforcement learning.  
"""

def delta():
	return self.r - self.v
	
def delta_dist(D):
	return self.r/D - self.v
