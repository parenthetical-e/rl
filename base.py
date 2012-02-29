from collections import defaultdict
""" 
The base is the Agent. But an Agent needs his attributes. You have to 
load em before you can go a' learning.
"""

class Agent(N):
	""" 
	Agent learns values and policies from errors.  We'll call this ethics. 
	
	If he is good he gets more rewards.
	
	Values and policies intersect right and wrong leading to errors.

	In space, right and wrong rest in one state or some other.  
	 
	Spaces don't exist till Agent arrives.  Next!
	
	Agent lives in Markov's world.  He is dumb.  He can almost never
	remember any history.
	
	When told to run, he runs.  But you have to tell him for how long.
	"""

	def __init__(self):
		self.N = None
		self.n = 0
		self.r = None
		self.rpe = None
		self.v = 0
		self.s = None
		self.sprime = None
		self.p_sprime = 0
		
		# Init and populate
		self.history = defaultdict(list)
		self.history['rpe'].append(self.rpe)
		self.history['r'].append(self.r)
		self.history['v'].append(self.v)
		self.history['s'].append(self.s)
		self.history['sprime'].append(self.sprime)
		self.history['p_sprime'].append(self.p_sprime)
		
		self.s_index = defaultdict(list)
		self.s_index[self.s].append(self.step_count)
	
	
	# Set attributes for each of the necessary RL functions
	# and their params
	def impose_ethics(self,v={'':{}},p={'':{}},e={'':{}}):
		""" Each has a name and some parameters """
		from rl.rlalgorithms import value,error,policy
		
		self.error_f = getattr(v.keys(),error)
		self.error_f_params = v.values()
		
		self.value_f = getattr(v.keys(),value)
		self.value_f_params = v.values()
		
		self.policy_f = getattr(p.keys(),policy)
		self.policy_f_params = p.values()
	
	
	def embed_into_environment(self,s={'':{}},r={'':{}}):		
		""" Embed the agent in state (s) and reward (r) spaces. """
		from rl.spaces import rewards,states
		
		self.state_space_f = getattr(s.keys(),states)
		self.state_space_f_params = s.values()
		
		self.reward_space_f = getattr(r.keys(),rewards)
		self.reward_space_f = r.values()
	

	def _update_history(self):
		""" Updates history with current data. """
		
		# 6. Update self.history with all
		# the new values (above).
		self.history['rpe'].append(self.rpe)
		self.history['r'].append(self.r)
		self.history['v'].append(self.v)
		self.history['s'].append(self.s)
		self.history['sprime'].append(self.sprime)
		self.history['p'].append(self.p_sprime)
	

	def _update_state_index(self):
		""" Updates the state index (and increments step_count). """
		self.s_index[self.s].append(self.step_count)
	
 
	def run(self,N):
		""" Learn by reinforcement. Run N steps. """

		self.N = N
		for self.n in range(self.N):
			
			# 1. Get new self.r self.s and restore self.v
			# based on current 's'
			self.r = self.reward_space.next()
			self.s = self.state_space.next()
			self.v = self.get_from_history(self,s,'v',1)	

			# 1.1. If s is null, zero everything and move on
			if (self.s is 0) or (self.s is '0'):
				self.rpe = 0
				self.r = 0
				self.v = 0
				self.s = 0
				self.sprime = None
				self.p_sprime = 0
				self._update_history()
				self._update_state_index()
				continue

			# 2. Calc rpe, update sefl.rpe
			self.error = self.error_f(**self.error_f_params)

			# 3. Calc value, update self.v
			self.v = self.value_f(**self.value_f_params)

			# 4. Pick next state (pretend to anyway, transitions 
			# are really define by [states]).
			self.sprime, self.p_sprime = self.policy_f(
					**self.policy_f_params)

			# 5. Update histroy and the state index
			self._update_history()
			self._update_state_index()		


	def get_from_history(self,s,name,num=1):
		"""
		Uses current state [s] and [name] of the data to retrieve the 
		last [num] entries from history.  
		
		[num] should be greater than 0.
		"""

		locations = self.s_index[s][-num]
		
		return self.history[name][locations]
