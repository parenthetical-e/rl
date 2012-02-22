""" A set of miscellaneous or helper functions for the rl package"""

def unpack(a_dict,states):
	"""
	For implementational simplicity, V, RPE, etc, values for all RL models 
	implemented here are stored in a dict, keyed on each state.  This 
	function uses states to unpack that dict into a list whose order 
	matches states.

	Note: Null (0) states are silently dropped.
	"""
	from copy import deepcopy
	
	a_dict_copy = deepcopy(a_dict)
		# otherwise the .pop() below 
		# would destroy a_dict
	a_list = []
	for s in states:
		if (s == 0) or (s == '0'):
			a_list.append(0.)
			continue
		else:
			a_list.append(a_dict_copy[s].pop(0))

	return a_list


def drop_null(data,states):
	"""Based on states, drop null states from a 1d data sequence."""
	
	no0 = []
	for d,s in zip(data,states):
		if (s == 0) or (s == '0'):
			continue
		else:
			no0.append(d)
	return no0
