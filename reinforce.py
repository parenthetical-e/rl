"""
Several reinforcement learning algorithms.

If they begin with a 'b_' they were implemented allow fitting of behavioral
accuracy data.  If they begin with a 's_' there were designed to simulate
'online' learning (e.g. a computational agent learning an N-armed bandit
task).
"""
import math

def b_delta(rewards,states,alpha):
	"""
	Implements the Resorla-Wagner (delta) learning rule.
	V_intial is 0.  Note: Null (0 or '0') states are silently skipped.

    Returns two dictionaries containing value and RPE timecourses, 
    for each state.
	"""
	
	# Init
	s_names = set(states)
	V_dict = {}
	RPE_dict = {}
	for s in s_names:
		V_dict[s] = [0.]
		RPE_dict[s] = []
	
	for r,s in zip(rewards,states):
		
		## Skip terminal states
		if (s == 0) | (s == '0'):
			continue
		
		V = V_dict[s][-1]
		
		## the Delta rule:
		RPE = r - V
		V_new = V + alpha * RPE
		
		## Store and shift V_new to
		## V for next iter
		V_dict[s].append(V_new)
		
		## Store RPE
		RPE_dict[s].append(RPE)
	
	return V_dict, RPE_dict


def b_delta_similarity(rewards,states,similarity,alpha):
	"""
	Implements the delta learning rule where the reward value is
	diminshed by the provded distances. V_intial is 0.  Note: Null
	(0 or '0') states are silently skipped.
	
	Reducing the reward value by the distance between a reward exmplar
	and its (possible) category representation is the subject of
	my dissertation research.

    Returns two dictionaries containing value and RPE timecourses, 
    for each state.
	"""

	# Init
	s_names = set(states)
	V_dict = {}
	RPE_dict = {}
    r_sim_dict = {}
	for s in s_names:
		V_dict[s] = [0.]
		RPE_dict[s] = []
        r_similarity[s] = []
	
	for r,s,d in zip(rewards,states,similarity):
		
		## Skip terminal states
		if (s == 0) | (s == '0'):
			continue
		
		V = V_dict[s][-1]
		
		## the Delta rule:
        r_sim = r * d
		RPE = r_sim - V
		V_new = V + alpha * RPE
		
        ## Store distance devalued reward
        
		## Store and shift V_new to
		## V for next iter
		V_dict[s].append(V_new)
		
		## Store RPE
		RPE_dict[s].append(RPE)
	
        ## Store r_sim
        r_sim_dict[s].append(r_sim)

	return V_dict, RPE_dict, r_sim_dict


def b_td_0(rewards,states,alpha):
	"""
	Implements Sutton and Barto's temporal difference algorithm, assuming
	gamma is 1. All V (values) initialized at zero.
	
	Arbitrary numbers of states are allowed; to simplify it was assumed
	that once started the markov process continues until the terminal state
	is achieved.
	
	Each trial is composed of one set of states which are contiguously
	packed separated only by null (0), that is to say terminal, states, the
	(empty) terminal state.
		
		Returns Qs a dict of lists and RPEs (list), in that order.
	"""
	
	## Taken form,
	## Sutton And Barto, Reinforcement Learning:
	## An Introduction, MIT Press, 1998,
	## TD is:
	## Intializa V(s) and pi (policy)
	## Repeat (for each trial)
	## 	Intializa s
	##  Repeat (for each step/state)
	##    a is the action given by pi for s
	##    Take a; observe reward (r) and the next state (s')
	##    V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))
	##    s <- s'
	##  until s is terminal
	
	gamma = 1
	
	## Init V_dict, and RPE_list
	s_names = list(set(states))
	V_dict = {}
	RPE_dict = {}
	for s in s_names:
		V_dict[s] = [0.]
		RPE_dict[s] = []
			# RPE should always
			# be n-1 compared to V
	
	for step in range(len(states)-1):
		r = rewards[step]
		s = states[step]
		s_plus = states[step+1]
		
		## Define values but then chck and
		## make sure were not in or before
		## a terminal state.
		V_s = V_dict[s][-1]
		V_s_plus = V_dict[s_plus][-1]
		
		if (s == 0) | (s == '0'):
			print '@NULL'
			continue
				# If we are terminal, terminate
		elif (s_plus == 0) | (s_plus == '0'):
			V_s_plus = 0
				# if the next state is terminal,
				# V_s_plus must be zero;
				# enforce that.
		
		print('step{0}, s{1}, s_plus{3}, r{2}'.format(step,s,r,s_plus))
		
		## And, finally, do the TD calculations.
		RPE = r + (gamma * (V_s_plus - V_s))
		V_s_new = V_s + (alpha * RPE)
		
		print '\t\tRPE {0}, V_s {1}'.format(RPE,V_s)
		V_dict[s].append(V_s_new)
		RPE_dict[s].append(RPE)
	
	return V_dict, RPE_dict


def b_rc(actions,rewards,beta):
	"""
	UNTESTED.
	
	Inplements Sutton And Barto's (1998) 'reinforcement comparison'
	algorithmn. Returning P(action) for each action at each timestep in a
	dict and the accompanying accumulative reference reward (i.e. the
	inline reward average) and reference predicion error (RPE).
	
	Beta is the step size (0-1).  Rewards in this model may be any real
	number (unlike sat most td implementations who are bound between 0-1).
	"""
	
	## In simulation action selection policy is set by softmax:
	## In this example there are three possible actions, extends to N
	## P(a_t = a) = e^P_t(a) / sum( e^P_t(b) + e^P_r(c) + ...)
	## Will display probability matching....
	
	ref_reward = 0
	P_dict = {}
	ref_reward_dict = {}
	RPE_dict = {}  # the reference prediction error (RPE)
	for a in set(actions):
		## Init P_dict, and the rest.
		## Start P_intial and ref_reward_dict as 0;
		## RPE is empty, keeping them in sink.
		P_dict[a] = [0]
		ref_reward_dict[a] = [0]
		RPE_dict[a] = []
	
	for a,r in actions,rewards:
		## Do calcs, then update dicts
		ref_reward = ref_reward_dict[a][-1]
		RPE = r - ref_reward
		P_a_tminus = P_dict[a][-1]
		P_a_t = P_a_tminus + (beta * RPE)
		
		P_dict[a] = P_dict[a].append(P_a_t)
		ref_reward_dict[a] = ref_reward_dict[a].\
				append((ref_reward + r) / 2)
		RPE_dict[a] = RPE_dict[a].append(RPE)
	
	return P_dict, ref_reward_dict, RPE_dict
