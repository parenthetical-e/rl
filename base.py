from collections import defaultdict

class RL(N):
	""" 
	A class for studying Reinforcement Learning algorithms.
	"""
	def __init__(self,N):
		self.N = N
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
	def set_rperror(self,alg,params):
		self.rperror = alg
		self.rperror_params = params
	
	
	def set_value(self,alg,params):
		self.value = alg
		self.value_params = params
	

	def set_policy(self,alg,params):
		self.policy = alg
		self.policy_params = params
	

	def set_state_space(self,gen,params):
		self.state_space = gen(**params)
			# On each iteration this should yield an update for self.s,
			# the current state
	

	def set_reward_space(self,gen,params):
		self.reward_space = gen(**params)
			# On each iteration this should yield an update for self.r,
			# the current reward
	

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
		""" Learn by reinforcement. Take N steps. """

		self.N = N
		for self.n in range(self.N):
			
			# 1. Get new self.r self.s and restore self.v
			# based on current 's'
			self.r = self.reward_space.next()
			self.s = self.state_space.next()
			self.v = self.get_from_history('v',1)	

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
			self.rpe = self.rperror(**self.rperror_params)

			# 3. Calc value, update self.v
			self.v = self.value(**self.value_params)

			# 4. Pick next state (pretend to anyway, transitions 
			# are really define by [states]).
			self.sprime, self.p_sprime = self.policy(**self.policy_params)

			# 5. Update histroy and the state index
			self._update_history()
			self._update_state_index()		
	

	def get_from_history(self,name,num=1):
		"""
		Uses current state [s] and [name] to retrieve the last [num]
		entries from history.
		"""

		if num == 0:
			num = 1
			## 0 makes no sense assume they intended 1

		locations = self.s_index[self.s][-num]
		return self.history[name][locations]
	
	def write_history(name='history.csv'):
		"""  """
		pass
	