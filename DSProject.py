### The same as SALEVELSTRAINTEST except with one level, using absrms as the error measure instead of the new cost function.

# SA functions including tracking testing set error at each step
## Same as SALevels except computes and records training error at each iteration as well.

# Buehler's sim annealing body, my individual functions, adapted for multiple levels.

### This file contains all functions necessary to run simulated annealing given a number of wavenumbers to choose (subset_size), 
### an initial temperature (T), the number of iterations to perform (iterations), and the spectral (spectral_data)
### and reference (reference_data) datasets.

### Algorithm adapted from Buehler, S. A., John, V. O., Kottayil, A., Milz, M., &#38; Eriksson, P. (2010). Efficient radiative transfer simulations for a broadband infrared radiometer-Combining a weighted mean of representative frequencies approach with frequency selection by simulated annealing. <i>Journal of Quantitative Spectroscopy and Radiative Transfer</i>, <i>111</i>(4), 602–615. https://doi.org/10.1016/j.jqsrt.2009.10.018

### Note: Data must be formatted in an xarray with wavenumbers, half-levels, and columns as dimensions

import xarray as xr
import numpy as np
import scipy.constants as sp
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# functions

def p(E, E_prime, T):
    # probability of accepting new state (Buehler 2010 eqn 6)
    return np.exp((E - E_prime)/T)
    
def rel_rms(estimate, reference):
    # relative root mean squared error across all ensembles
    # Buehler 2010 eqn 3
    return np.sqrt((((estimate - reference)/reference)**2).mean(axis = 1))
    
def abs_rms(estimate, reference):
    # absolute root mean squared error across all ensembles
    # Buehler 2010 eqn 3
    return np.sqrt(((estimate - reference.data)**2).mean(axis = 1))

def cost(estimate, reference):
    #error = rel_rms(estimate, reference)
    #return np.sqrt(((error)**2).mean(axis = 0))
    ### TODO: CHANGED THIS TO ABSRMS FOR THE PROJECT 
    return abs_rms(estimate, reference)
    
def neighbor(S, num_wavenumbers):
    # takes in state S, returns state S' where one random frequency in S is replaced by a random inactive frequency
    # S is a vector of subset_size wavenumber indeces
    # Buehler 2010 eqn 5
    S_prime = np.array(S)
    # choose new wavenumber
    new_idx = random.randint(0, num_wavenumbers - 1)
    # make sure new wavenumber doesn't already exist in S
    exists = new_idx in S
    while exists == True:
        new_idx = random.randint(0, num_wavenumbers - 1)
        exists = new_idx in S
    # choose an index in S to replace with the new wavenumber
    old_idx = random.randint(0, len(S) - 1)
    S_prime[old_idx] = new_idx
    return np.array(S_prime)

def compute_q_tilde(S, spectral_data, reference_data, num_hl, num_cols):
    # TO DO : HARDCODED FOR 1 LEVEL
    q_v = spectral_data.isel(wavenumber = S).data
    S_original = S
    
    no_negatives = False
    while(no_negatives == False):

        #q_tilde = np.empty((num_hl, num_cols))
        w = np.empty((num_hl, len(S)))

        for i in range(num_hl):
            #clf = LinearRegression()
            clf = Ridge(alpha = 10**(-5))
            clf.fit(spectral_data.isel(wavenumber = S, half_level = i).data, reference_data.isel(half_level = i).data)
            w[i] = clf.coef_
            
        remove = w.max(axis = 0) < 0
        w[:, remove] = 0
        S = np.delete(S, remove)
        if(not(remove.any() == True)): 
            no_negatives = True

    w_final = np.zeros((num_hl, len(S_original)))
                       
    for h in range(num_hl):
        for n in range(len(S)):
            for i in range(len(S_original)):
                if (S[n] == S_original[i]):
                    w_final[h][i] = w[h][n]

            
    q_tilde = clf.predict(spectral_data.isel(wavenumber = S, half_level = 0).data)
                       
    return w_final, clf.intercept_, q_v, q_tilde

def compute_testing(S, w, intercept, spectral_test, reference_test, num_hl, num_cols):
    #q_test = np.empty((num_hl, num_cols))
    
    # compute estimate using weights and S from current training estimate but with spectral data from testing test
    q_test = np.matmul(w, spectral_test.isel(wavenumber = S, half_level = 0).data.T) + intercept
        
    # compute error of test estimate against test reference calculation
    error = abs_rms(q_test, reference_test)
    
    return error

def anneal(spectral_data, reference_data, n, block_size, spectral_test, reference_test):
    N_succ = 100 # number successful cases needed in a block
    T_fact = 0.9 # factor by which we decrease temp
    num_hl = len(reference_data.half_level.data)
    num_cols = len(reference_data.column.data)
    num_wavenumbers = len(spectral_data.wavenumber.data)

    # initial guess
    S = np.array(random.sample(range(num_wavenumbers - 1), k = n))
    sb = S
    #w, q_v, q_tilde = compute_q_tilde(S, spectral_data, reference_data, num_hl, num_cols)
    #E = abs_rms(q_tilde, reference_data, num_hl)

    # history in one block
    cost_shorthist = np.zeros(block_size)
    e_shorthist = np.zeros((num_hl, block_size))
    e_testshort = np.zeros((num_hl, block_size)) # keeping track of testing history
    
    # initial temp as in Buehler et al
    for i in range(block_size):
        sn = neighbor(S, num_wavenumbers)
        w, intercept, q_v, q_tilde = compute_q_tilde(sn, spectral_data, reference_data, num_hl, num_cols)
        e = abs_rms(q_tilde, reference_data)
        e_test = compute_testing(sn, w, intercept, spectral_test, reference_test, num_hl, num_cols)
        c = cost(q_tilde, reference_data)
        cost_shorthist[i] = c
        e_shorthist[:, i] = e
        e_testshort[:, i] = e_test
        S = np.array(sn)
    
    # From Vicente et al.
    # We want to sete Tstart such that at the beginning almost all
    # moves are accepted. (Divide by ln(acceptance probability).)
    T = -np.mean(np.abs(np.diff(cost_shorthist)))/np.log(0.99)
    
    t_hist     = np.empty(1000)
    cost_hist  = np.empty(1000)
    c_std_hist = np.empty(1000)
    c_min_hist = np.empty(1000)
    c_max_hist = np.empty(1000) 
        
    # INDIVIDUAL ERRORS
    e_hist = np.zeros((num_hl, 1000))
    e_testhist = np.zeros((num_hl, 1000))
    
    Wb = np.zeros(len(S))
    t_hist[0]     = T
    cost_hist[0]     = np.mean(cost_shorthist)
    c_std_hist[0] = np.std(cost_shorthist)
    c_min_hist[0] = np.min(cost_shorthist)
    c_max_hist[0] = np.max(cost_shorthist)
    e_hist[:, 0] = np.mean(e_shorthist, axis = 1)
    e_testhist[:, 0] = np.mean(e_testshort, axis = 1)

    cb = c_max_hist[0]

    k = 0

    go_on = True
    while(go_on == True):
        n_succ = 0 
        for i in range(block_size):
            S_prime = neighbor(S, num_wavenumbers)
            w_prime, intercept_prime, q_v_prime, q_tilde_prime = compute_q_tilde(S_prime, spectral_data, reference_data, num_hl, num_cols)
            e_prime = abs_rms(q_tilde_prime, reference_data)
            e_test_prime = compute_testing(S_prime, w_prime, intercept_prime, spectral_test, reference_test, num_hl, num_cols)
            c_prime = cost(q_tilde_prime, reference_data)
            de = c_prime - c
            
            if (c_prime < cb):
                sb = np.array(S_prime)
                Wb = w_prime
                cb = c_prime
            
            if (random.random() < p(c, c_prime, T)):
                S = S_prime
                c = c_prime
                e = e_prime
                e_test = e_test_prime
                n_succ = n_succ + 1
                
            cost_shorthist[i] = c
            e_shorthist[:, i] = e
            e_testshort[:, i] = e_test
            
            if (n_succ >= N_succ):
                break
            
        k = k + 1
        t_hist[k]     = T
        cost_hist[k]     = np.mean(cost_shorthist[0 : i + 1])
        c_std_hist[k] = np.std(cost_shorthist[0 : i + 1])
        c_min_hist[k] = np.min(cost_shorthist[0 : i + 1])
        c_max_hist[k] = np.max(cost_shorthist[0 : i + 1])
        e_hist[:, k]     = np.mean(e_shorthist[:, :i + 1], axis = 1)
        e_testhist[:, k] = np.mean(e_testshort[:, :i + 1], axis = 1)
          
        if (cost_hist[k] >= cost_hist[k - 1]):
            T = T * T_fact

        # Should we continue? - We use as stop criterion that there were
        # no more successful moves
        if (n_succ == 0):
            go_on = False
            
        # SEE WHAT HAPPENS AFTER 999 ITERATIONS # TODO?
        if (k > 800):
            go_on = False
            
    # cut off NANs from returnables
    return sb, Wb, cb, t_hist[:k + 1], cost_hist[:k + 1], e_hist[:, :k + 1], e_testhist[:, :k + 1]

# main

def sim_loop(n_start, block_size, spectral_data, reference_data, accuracy, spectral_test, reference_test):
    # like buehler but increment default 1
    go_on = True
    i = 0
    n = n_start
    
    s_best, W_best, cost_best, t_hist, cost_hist, e_hist, e_testhist = anneal(spectral_data, reference_data, n, block_size, spectral_test, reference_test)
    
    #if (cost_best <= accuracy):
    go_on = False
    #else:
    #    i = i + 1
    #    n = n + n_incr # TODO: n_incr is undefined but that works bc I'm not incrementing rn

    while (go_on == True):
        sb, Wb, eb, th, ch, eh, et = anneal(spectral_data, reference_data, n, block_size, spectral_test, reference_test)

        s_best = np.append(s_best, sb)
        W_best = np.append(W_best, np.array(Wb), axis = 0)
        cost_best = np.append(cost_best, eb)
        t_hist = np.append(t_hist, th)
        cost_hist = np.append(cost_hist, ch)
        e_hist = np.append(e_hist, np.array(eh), axis = 0)
        e_testhist = np.append(e_testhist, np.array(et), axis = 0)
        
        if (cost_best <= accuracy):
            go_on = False
        else:
            i = i + 1
            n = n + n_incr # TODO: n_incr is undefined but that works bc I'm not incrementing rn

    # compute error for best set
    w, intercept, q_v, q_tilde = compute_q_tilde(s_best, spectral_data, reference_data, len(reference_data.half_level.data), len(reference_data.column.data))
    e_best = abs_rms(q_tilde, reference_data)
    
    return s_best, W_best, cost_best, t_hist, cost_hist, e_best, e_hist, e_testhist

