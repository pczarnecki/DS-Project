import xarray as xr
import numpy as np
import random


def data_setup(level_array, level_type, ref_input, spec_input, heating_data = None):
    cp = 1004 #specific heat
    g = 9.81 #gravity

    # open summed up reference data
    ref_data = xr.open_dataset(ref_input,
                         engine = "netcdf4")

    # Spectral flux data - fluxes per wavenumber
    spec_fluxes = xr.open_mfdataset(spec_input, 
                                    combine = 'nested', concat_dim = 'column',
                                   engine = "netcdf4")

    # set up coordinates/arrays
    wavenumber_coords = spec_fluxes.wavenumber.data

    num_cols = min(len(ref_data.column.data), len(spec_fluxes.column.data))
    
    ref = np.empty((len(level_array), num_cols))
    flux_subset = np.empty((len(level_array), num_cols, len(spec_fluxes.wavenumber.data)))

    for i in range(len(level_array)):
        if (level_type[i] == 'f'):
            if (level_array[i] == 0):
                # TOA outgoing fluxes
                TOA_up_spec = spec_fluxes.spectral_flux_up_lw.isel(half_level = 0).data
                TOA_up_ref = TOA_up_spec.sum(axis = 1)
                ref[i] = TOA_up_ref.compute()

                TOA_up_spec = TOA_up_spec.compute() 
                TOA_up_spec = np.array(TOA_up_spec)
                flux_subset[i] = TOA_up_spec

            elif (level_array[i] == 54):
                # ground downward fluxes
                BOA_dn_spec = spec_fluxes.spectral_flux_dn_lw.isel(half_level = 54).data
                BOA_dn_ref = BOA_dn_spec.sum(axis = 1)
                ref[i] = BOA_dn_ref.compute()

                BOA_dn_spec = BOA_dn_spec.compute()
                BOA_dn_spec = np.array(BOA_dn_spec)
                flux_subset[i] = BOA_dn_spec

            else:
                # net flux in interior levels
                lev_spec = (spec_fluxes.spectral_flux_dn_lw.isel(half_level = level_array[i]).data - 
                            spec_fluxes.spectral_flux_up_lw.isel(half_level = level_array[i]).data)
                flux_subset[i] = lev_spec
                ref[i] = lev_spec.sum(axis = 1)

        elif (level_type[i] == 'h'):
            # interior heating rates
            lev_spec = (spec_fluxes.spectral_flux_dn_lw.isel(half_level = level_array[i]).data - 
                        spec_fluxes.spectral_flux_up_lw.isel(half_level = level_array[i]).data)
            flux_subset[i] = lev_spec
            
            net_ref = (ref_data.flux_dn_lw.isel(half_level = level_array[i]).data - 
                        ref_data.flux_up_lw.isel(half_level = level_array[i]).data)
            net_ref_up = (ref_data.flux_dn_lw.isel(half_level = level_array[i] + 1).data - 
                        ref_data.flux_up_lw.isel(half_level = level_array[i] + 1).data)
            dF = net_ref_up - net_ref
            dp = ref_data.pressure_hl.isel(half_level = level_array[i] + 1).data - ref_data.pressure_hl.isel(half_level = 
                                                                                                         level_array[i]).data
            ref[i] = -(g/cp)*(dF/dp)

        else:
            print("level identifier must be either f(lux) or h(eating rate)")

    # assemble data into xarrays
    ref = xr.DataArray(data = ref, dims = ["half_level", "column"], 
                      coords = dict(half_level=(["half_level"], level_array)))

    flux_subset = xr.DataArray(data = flux_subset, dims = ["half_level", "column", "wavenumber"],
                              coords = dict(half_level=(["half_level"], level_array), 
                                            wavenumber=(["wavenumber"], wavenumber_coords)))

    return flux_subset, ref

def data_setup_wrong(level_array, level_type, ref_input, spec_input, heating_data = None):
    # open summed up reference data
    ref_data = xr.open_dataset(ref_input,
                         engine = "netcdf4")

    # Spectral flux data - fluxes per wavenumber
    spec_fluxes = xr.open_mfdataset(spec_input, 
                                    combine = 'nested', concat_dim = 'column',
                                   engine = "netcdf4")
    if (any(level_type == 'h')):
        # monochromatic heating
        heating = heating_data
        heating = heating.transpose("column", "half_level", "wavenumber")

        # heating = xr.open_dataset("/dx02/pc2943/mono_heating.h5", engine = "netcdf4")
        # heating = heating.rename({'mul-586aef02731ae840e3f7c86cffab3ac2':'mono_heating'})

    # set up coordinates/arrays
    wavenumber_coords = spec_fluxes.wavenumber.data

    num_cols = min(len(ref_data.column.data), len(spec_fluxes.column.data))
    
    ref = np.empty((len(level_array), num_cols))
    flux_subset = np.empty((len(level_array), num_cols, len(spec_fluxes.wavenumber.data)))

    for i in range(len(level_array)):
        if (level_type[i] == 'f'):
            if (level_array[i] == 0):
                # TOA outgoing fluxes
                TOA_up_spec = spec_fluxes.spectral_flux_up_lw.isel(half_level = 0).data
                TOA_up_ref = TOA_up_spec.sum(axis = 1)
                ref[i] = TOA_up_ref.compute()

                TOA_up_spec = TOA_up_spec.compute() 
                TOA_up_spec = np.array(TOA_up_spec)
                flux_subset[i] = TOA_up_spec

            elif (level_array[i] == 54):
                # ground downward fluxes
                BOA_dn_spec = spec_fluxes.spectral_flux_dn_lw.isel(half_level = 54).data
                BOA_dn_ref = BOA_dn_spec.sum(axis = 1)
                ref[i] = BOA_dn_ref.compute()

                BOA_dn_spec = BOA_dn_spec.compute()
                BOA_dn_spec = np.array(BOA_dn_spec)
                flux_subset[i] = BOA_dn_spec

            else:
                # net flux in interior levels
                lev_spec = (spec_fluxes.spectral_flux_dn_lw.isel(half_level = level_array[i]).data - 
                            spec_fluxes.spectral_flux_up_lw.isel(half_level = level_array[i]).data)
                flux_subset[i] = lev_spec
                ref[i] = lev_spec.sum(axis = 1)

        elif (level_type[i] == 'h'):
            # interior heating rates
            lev_spec = heating.mono_heating.isel(half_level = level_array[i]).data
            flux_subset[i] = lev_spec
            ref[i] = lev_spec.sum(axis = 1)

        else:
            print("level identifier must be either f(lux) or h(eating rate)")

    # assemble data into xarrays
    ref = xr.DataArray(data = ref, dims = ["half_level", "column"], 
                      coords = dict(half_level=(["half_level"], level_array)))

    flux_subset = xr.DataArray(data = flux_subset, dims = ["half_level", "column", "wavenumber"],
                              coords = dict(half_level=(["half_level"], level_array), 
                                            wavenumber=(["wavenumber"], wavenumber_coords)))

    return flux_subset, ref

def data_setup_train_test(level_array, level_type, spec_input, ref_input, num_train, heating_data = None):
    # open summed up reference data
    ref_data = xr.open_mfdataset(ref_input,
                                 combine = 'nested', concat_dim = 'column',
                         engine = "netcdf4")

    # Spectral flux data - fluxes per wavenumber
    spec_fluxes = xr.open_mfdataset(spec_input, 
                                    combine = 'nested', concat_dim = 'column',
                                   engine = "netcdf4")
        
    if (any(level_type == 'h')):
        # monochromatic heating
        heating = heating_data
        heating = heating.transpose("column", "half_level", "wavenumber")

        # heating = xr.open_dataset("/dx02/pc2943/mono_heating.h5", engine = "netcdf4")
        # heating = heating.rename({'mul-586aef02731ae840e3f7c86cffab3ac2':'mono_heating'})

    # set up coordinates/arrays
    wavenumber_coords = spec_fluxes.wavenumber.data

    num_cols = min(len(ref_data.column.data), len(spec_fluxes.column.data))
    
    ref = np.empty((len(level_array), num_cols))
    flux_subset = np.empty((len(level_array), num_cols, len(spec_fluxes.wavenumber.data)))
    
    num_test = num_cols - num_train

    for i in range(len(level_array)):
        if (level_type[i] == 'f'):
            if (level_array[i] == 0):
                # TOA outgoing fluxes
                TOA_up_spec = spec_fluxes.spectral_flux_up_lw.isel(half_level = 0).data
                TOA_up_ref = TOA_up_spec.sum(axis = 1)
                ref[i] = TOA_up_ref.compute()

                TOA_up_spec = TOA_up_spec.compute() 
                TOA_up_spec = np.array(TOA_up_spec)
                flux_subset[i] = TOA_up_spec

            elif (level_array[i] == 54):
                # ground downward fluxes
                BOA_dn_spec = spec_fluxes.spectral_flux_dn_lw.isel(half_level = 54).data
                BOA_dn_ref = BOA_dn_spec.sum(axis = 1)
                ref[i] = BOA_dn_ref.compute()

                BOA_dn_spec = BOA_dn_spec.compute()
                BOA_dn_spec = np.array(BOA_dn_spec)
                flux_subset[i] = BOA_dn_spec

            else:
                # net flux in interior levels
                lev_spec = (spec_fluxes.spectral_flux_dn_lw.isel(half_level = level_array[i]).data - 
                            spec_fluxes.spectral_flux_up_lw.isel(half_level = level_array[i]).data)
                flux_subset[i] = lev_spec
                ref[i] = lev_spec.sum(axis = 1)

        elif (level_type[i] == 'h'):
            # interior heating rates
            lev_spec = heating.mono_heating.isel(half_level = level_array[i]).data
            flux_subset[i] = lev_spec
            ref[i] = lev_spec.sum(axis = 1)

        else:
            print("level identifier must be either f(lux) or h(eating rate)")

    # choose training and testing subsets
    train_set = np.sort(np.array(random.sample(range(num_cols - 1), k = num_train)))
    test_set = np.arange(num_cols)
    test_set = np.delete(test_set, train_set)
    
    # assemble data into xarrays
    ref_train = xr.DataArray(data = ref[:, train_set], dims = ["half_level", "column"], 
                      coords = dict(half_level=(["half_level"], level_array), column = (["column"], train_set)))
    ref_test = xr.DataArray(data = ref[:, test_set], dims = ["half_level", "column"], 
                      coords = dict(half_level=(["half_level"], level_array), column = (["column"], test_set)))

    spec_train = xr.DataArray(data = flux_subset[:, train_set, :], dims = ["half_level", "column", "wavenumber"],
                              coords = dict(half_level=(["half_level"], level_array), 
                                            wavenumber=(["wavenumber"], wavenumber_coords),
                                            column =(["column"], train_set)))
    spec_test = xr.DataArray(data = flux_subset[:, test_set, :], dims = ["half_level", "column", "wavenumber"],
                          coords = dict(half_level=(["half_level"], level_array), 
                                        wavenumber=(["wavenumber"], wavenumber_coords),
                                        column =(["column"], test_set)))


    return ref_train, ref_test, spec_train, spec_test


def demean_and_normalize(train, test):
    # expects spectral dataset
    mean = train.mean(dim = 'column')
    std = train.std(dim = 'column')
    new_train = (train - mean)/(std**2)
    new_test = (test - mean)/(std**2)
    return new_train, new_test
