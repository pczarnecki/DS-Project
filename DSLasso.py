### Find an appropriate alpha for LASSO regression to get a subset size of about 60
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import scipy.constants as sp
import  DataSetup as DS
import SADSProject as SA
from sklearn.linear_model import Lasso

### Given a wavenumber subset, weights from the model, and data, compute the error the model produces
#def compute_estimate_error(model, S, spectral_data, reference_data):
#    q_tilde = model.predict(spectral_data.data[0])
#    error = SA.abs_rms(q_tilde, reference_data)
#    return error


### Set up testing and training data
level_array = np.array([0]) # which hls to use
level_type = np.array(['f']) # is the corresponding level heating rate or radiative flux?
ref_input = "/dx02/robertp/CKDMIP_LBL/evaluation1/lw_fluxes/ckdmip_evaluation1_lw_fluxes_present.h5"
spec_input = ["/dx02/pc2943/spectral_fluxes_1-10.h5", "/dx02/pc2943/spectral_fluxes_11-20.h5",
                                "/dx02/pc2943/spectral_fluxes_21-30.h5", "/dx02/pc2943/spectral_fluxes_31-40.h5",
                                "/dx02/pc2943/spectral_fluxes_41-50.h5"]
specTest_input = ["/dx02/pc2943/eval2_spectral_fluxes_1-10.h5", "/dx02/pc2943/eval2_spectral_fluxes_11-20.h5",
                                "/dx02/pc2943/eval2_spectral_fluxes_21-30.h5", "/dx02/pc2943/eval2_spectral_fluxes_31-40.h5",
                                "/dx02/pc2943/eval2_spectral_fluxes_41-50.h5"]
refTest_input = "/dx02/robertp/CKDMIP_LBL/evaluation2/lw_fluxes/ckdmip_evaluation2_lw_fluxes-4angle_present.h5"

flux_subset, ref = DS.data_setup(level_array, level_type, ref_input, spec_input)
spec_test, ref_test = DS.data_setup(level_array, level_type, refTest_input, specTest_input)
num_hl = 1
num_cols = 50

# initial values
alpha_val = 5*10**(-5)
S_index = np.array([np.empty(1000)])

# first iteration
model = Lasso(alpha = alpha_val, positive = True)
model = model.fit(flux_subset.data[0], ref.data[0])
S_index = np.where(model.coef_ > 0)

## keep track of different values
alpha_list = alpha_val
S_list = len(S_index[0])

# compute testing/training error
#train_errors = compute_estimate_error(model, S_index[0], flux_subset, ref)
#test_errors = compute_estimate_error(model, S_index[0], spec_test, ref_test)
w_final, intercept, q_v, q_tilde = SA.compute_q_tilde(S_index[0], flux_subset, ref, num_hl, num_cols)
train_errors = SA.abs_rms(q_tilde, ref)
test_errors = SA.compute_testing(S_index[0], w_final, intercept, spec_test, ref_test, num_hl, num_cols)


## loop through different alpha values until we get to a subset size of about 50
while (len(S_index[0]) > 50):
    alpha_val = alpha_val + (1 * 10**(-4))
    model = Lasso(alpha = alpha_val, positive = True)
    model = model.fit(flux_subset.data[0], ref.data[0])
    S_index = np.where(model.coef_ > 0)
    
    # record values
    alpha_list = np.append(alpha_list, alpha_val)
    S_list = np.append(S_list, len(S_index[0]))
    #train_errors = np.append(train_errors, compute_estimate_error(model, S_index[0], flux_subset, ref))
    #test_errors = np.append(test_errors, compute_estimate_error(model, S_index[0], spec_test, ref_test))
    
    w_final, intercept, q_v, q_tilde = SA.compute_q_tilde(S_index[0], flux_subset, ref, num_hl, num_cols)
    train_errors = np.append(train_errors, SA.abs_rms(q_tilde, ref))
    test_errors = np.append(test_errors, SA.compute_testing(S_index[0], w_final, intercept, spec_test, ref_test, num_hl, num_cols))

    print("alpha = ", alpha_val) # keep track of where we are
    print("subset size = ", len(S_index[0]))

## save all our data
results = xr.Dataset(
    data_vars=dict(
        train_errors=(["alpha_list"], train_errors),
        test_errors=(["alpha_list"], test_errors),
        subset_size = S_list
    ),
    coords = dict(
        alpha_list = alpha_list,
    )
)

results.to_netcdf('DS_Lasso_LinearFit_intercept.h5')