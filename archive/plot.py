"""
A set of functions for visualizing fmri and behavoral simulations. Requires
matplotlib. 
"""
import matplotlib.pyplot as plt

def _acc(x,acc,ax):
	"""
	Draws a pretty barplot pf accuracy data. If x is None a 0 indexed 
	list of length acc is used.  Returns the plot object.
	"""
	if x is None:
		x = range(len(acc))

	try:	
		p_a = ax.bar(x,acc,label='acc',color='grey',alpha=.3)
	except NameError:
		raise('No figure is defined at {0}'.format(ax))

	return p_a


def _p(x,p,conditions,ax):
	"""
	Creates a line plot of probability data in blue.  A grey line 
	indicating chance, as determind by the number of unique conditons
	(excluding Nulls), is added.  If x is None a 0 indexed list of length 
	acc is used.  Returns the plot object.
	"""

	if x is None:
		x = range(len(p))
	
	if 0 in set(conditons):
		chance = 1./len(set(conditons))-1
	else:
		chance = 1./len(set(conditons))
	try:	
		p_horiz = ax.plot(
				chance*len(p_2),color='black',linewidth=1.5,alpha=.3)
		p_p = ax.plot(p_2,label='p',color='blue',alpha=.5,linewidth=2) 
	except NameError:
		raise('No figure is defined at {0}'.format(ax))

	return p_p,p_horiz


def _rl(x,values,rpes,ax):
	"""
	Creates a line plot of values (green) and reward prediction errors (red)
	If x is None a 0 indexed list of length acc is used.  Returns the 
	plot objects for each.
	"""

	if x is None:
		x = range(len(p))

	try:
		p_v = ax.plot(x,values,label='values',color='orange',linewidth=2,alpha=.6)
		p_rpe = ax.plot(rpes,label='RPE',color='red',linewidth = 2,alpha=.6)
	except NameError:
		raise('No figure is defined at {0}'.format(ax))

	return p_v,p_rpe


def by_conditions(conditions,values):
	"""
	Plot each entry in datasets (a dict keyed on names of the datasets, the 
	values bieng an iteraratable sequnce of numbers) and make a seperate 
	plot for every condition

	...and sometimes magical pretty things happen.
	"""
	
	c_names = sort(set(conditions))

	# plot init
	fig = plt.figure()
	plt.subplot(len(c_names),1,1) # n rows is the subplot
										  # number of conditions
	for ii,c_n in enumerate(c_names):
		plt.subplot(1,1,ii)
		for d_name,dataset in datasets.items():
		
			## Specal cases: 
			## Is it accuracy data,if so do sumpin' pretty
			if d_name is 'acc':
				bar(range(len(dataset)),dataset,color='grey',alpha=.3)
		
			##  
		
	
