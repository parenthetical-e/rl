"""Fit parameters to behavoiral data"""
import numpy as np
import rl


def ml_delta(acc,states,res):
    """
    Use maximum likelihood to find the best alpha and beta values for the 
    delta learning rule.
    """
    from itertools import product
    from copy import deepcopy

    params = ((0.01,1),(0.01,5))
        # alpha, beta ranges

    param_values = [np.arange(*par,step=res) for par in params]
    all_unique_param = product(*param_values)
        # http://www.technovelty.org/code/python/asterisk.html
        # for a nice tutorial on asterices and python
    
    best_log_L = None
    best_par = ()

    for alpha,beta in all_unique_param:
        v_dict, rpe_dict = rl.reinforce.b_delta(acc,states,alpha)
        v = rl.misc.unpack(v_dict,states)

        p_values = np.array(rl.policy.softmax(v,states,beta))
        log_L = np.sum(np.log(p_values))
        
        if log_L > best_log_L:
            print('Improvement - L:{0}, alpha:{1}, beta:{2}'.format(
                log_L,alpha,beta))
            best_log_L = deepcopy(log_L)
            best_par = (alpha,beta)

    return (best_par,best_log_L)
    

def ml_delta_similarity(acc,states,similarity,res):
    """
    Use maximum likelihood to find the best alpha and beta values for the 
    similarity-adjusted delta learning rule.
    """
    from itertools import product
    from copy import deepcopy

    params = ((0.01,1),(0.01,5))
        # alpha, beta ranges

    param_values = [np.arange(*par,step=res) for par in params]
    all_unique_param = product(*param_values)
        # http://www.technovelty.org/code/python/asterisk.html
        # for a nice tutorial on asterices and python

    best_log_L = None
    best_par = ()

    for alpha,beta in all_unique_param:
        v_dict, rpe_dict, acc_sim = rl.reinforce.b_delta_similarity(
                acc,states,similarity,alpha)
        v = rl.misc.unpack(v_dict,states)

        p_values = np.array(rl.policy.softmax(v,states,beta))
        log_L = np.sum(np.log(p_values))

        if log_L > best_log_L:
            print('Improvement - L:{0}, alpha:{1}, beta:{2}'.format(
                log_L,alpha,beta))
            best_log_L = deepcopy(log_L)
            best_par = (alpha,beta)

    return (best_par,best_log_L)
