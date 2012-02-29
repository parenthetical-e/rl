def flatten(rewards,states):
	""" Flattens (2d) reward and state space representations. """
	
	# Init the returned and ensure they match
	if rewards.shape != states.shape:
		raise ValueError('rewards and states have different number of rows')
	
	flat_states = zeros(states.shape[0])
	for ii,s in enumerate(range(1,states.shape[1]+1)):
		flat_states[states[...,ii] == 1] = s
	
	flat_rewards = rewards.mean(1)
	if sum(flat_rewards > 1):
		raise ValueError('rewards contained more than one "1" per row.')
			# reward should only have one '1' per row
	
	return flat_rewards,flat_states
