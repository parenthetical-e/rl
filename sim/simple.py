""" 
This is really simple top-level run file for a RL experiment using Agent.
"""
from rl.base import Agent

# What will his ethics be?
error={}
value={}
policy={}

# Define the environment
states={}
rewards={}

# Make an Agent. Name him Tim.
tim = Agent()

# Set him up.
tim.impose_ethics(error,value,policy)
tim.embed_into_environment(states,rewards)

# Run Tim run.  Learn 100 iterations!
tim.run(100)

# His history are your results
results = tim.history