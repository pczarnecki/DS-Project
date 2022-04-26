import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import scipy.constants as sp
import random
from sklearn.linear_model import LinearRegression
import itertools
from scipy.integrate import quad
from scipy.optimize import root_scalar
import DataSetup as DS
import DSProject as SA

#set up data
ref_input = "/data/robertp/CKDMIP_LBL/evaluation1/lw_fluxes/ckdmip_evaluation1_lw_fluxes_present.h5"
spec_input = ["/data/pc2943/spectral_fluxes_1-10.h5", "/data/pc2943/spectral_fluxes_11-20.h5",
                                "/data/pc2943/spectral_fluxes_21-30.h5", "/data/pc2943/spectral_fluxes_31-40.h5",
                                "/data/pc2943/spectral_fluxes_41-50.h5"]
specTest_input = ["/data/pc2943/eval2_spectral_fluxes_1-10.h5", "/data/pc2943/eval2_spectral_fluxes_11-20.h5",
                                "/data/pc2943/eval2_spectral_fluxes_21-30.h5", "/data/pc2943/eval2_spectral_fluxes_31-40.h5",
                                "/data/pc2943/eval2_spectral_fluxes_41-50.h5"]
refTest_input = "/data/robertp/CKDMIP_LBL/evaluation2/lw_fluxes/ckdmip_evaluation2_lw_fluxes-4angle_present.h5"

#level_array = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 54]) # which hls to use
#level_type = np.array(['f', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'f']) # is the corresponding level heating rate or radiative flux?
level_array = np.array([0])
level_type = np.array(['f'])
#num_train = 80

#level_array = np.arange(55)
#level_type = np.full(55, 'h')
#level_type[0] = 'f'
#level_type[-1] = 'f'


#ref_train, ref_test, spec_train, spec_test = DS.data_setup_train_test(level_array, level_type, spec_input, ref_input, num_train, None)
spec_train, ref_train = DS.data_setup(level_array, level_type, ref_input, spec_input, None)
spec_test, ref_test = DS.data_setup(level_array, level_type, refTest_input, specTest_input, None)

## save dataset
#spec_train.to_netcdf('/data/pc2943/eval1_spec_every5levels_data.h5')
#ref_train.to_netcdf('/data/pc2943/eval1_ref_every5levels_data.h5')
#spec_test.to_netcdf('/data/pc2943/eval2_spec_every5levels_data.h5')
#ref_test.to_netcdf('/data/pc2943/eval2_ref_every5levels_data.h5')

### Loading data from file
#spec_train = xr.open_dataarray('/data/pc2943/eval1_spec_every5levels_data.h5', engine = "netcdf4")
#ref_train = xr.open_dataarray('/data/pc2943/eval1_ref_every5levels_data.h5', engine = "netcdf4")
#spec_test = xr.open_dataarray('/data/pc2943/eval2_spec_every5levels_data.h5', engine = "netcdf4")
#ref_test = xr.open_dataarray('/data/pc2943/eval2_ref_every5levels_data.h5', engine = "netcdf4")


# normalize and demean
#spec_train, spec_test = DS.demean_and_normalize(spec_train, spec_test)

# define params for running
n_start = 100
block_size = 1000
accuracy = 10

# run
#s_best, W_best, e_best, T_hist, e_hist = SA.sim_loop(n_start, block_size, spec_train, ref, accuracy)
s_best, W_best, cost_best, T_hist, cost_hist, e_best, e_hist, e_test = SA.sim_loop(n_start, block_size, spec_train, ref_train, accuracy, spec_test, ref_test) # mine
#s_best, W_best, E_best, T_hist, E_hist = SA.loop_anneal(spec_train, ref, n_start, 1, accuracy) # Buehler's

# saving
blocks = np.arange(len(T_hist))

results = xr.Dataset(
    data_vars=dict(
        cost_hist=(["blocks"], cost_hist),
        e_hist =(["half_levels", "blocks"], e_hist),
        e_test =(["half_levels", "blocks"], e_test),
        temps=(["blocks"], T_hist),
        W = (["S"], W_best[0]),
        E = e_best,
        C = cost_best,
    ),
    coords = dict(
    blocks = blocks,
    half_levels = spec_train.half_level.data,
    S = s_best,
    ),
    attrs=dict(description="rerunning the ridge regression with alpha = 10-5, with 60 rep wavenumbers."),
) 

#results = xr.Dataset(
#    data_vars=dict(
#        e_hist =(["blocks"], e_hist),
#        temps=(["blocks"], T_hist),
#        W = (["S"], W_best),
#        E = e_best,
#    ),
#    coords = dict(
#    blocks = blocks,
#    half_levels = spec_train.half_level.data,
#    S = s_best,
#    ),
#    attrs=dict(description="TOA with 60 points to see if original SAFunctions code produces same result as newer code."),
#)


results.to_netcdf('Linear_100wn_shorterepoch.h5')