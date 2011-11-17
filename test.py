import rl
import simBehave
import simfMRI

def bleep():
	tr,ac,pa = simBehave.behave.all_learn(3,10,3,True,False)
	print(tr)
	print(ac)
	print(pa)
	
	hh = simfMRI.hrf.double_gamma(30)
	print(hh)

	v, rpe = rl.reinforce.b_delta(ac,tr,.3)
	print(v)
	print(rpe)

	p = rl.policy.softmax(rl.misc.unpack(v,tr),tr,3)
	print(p)
