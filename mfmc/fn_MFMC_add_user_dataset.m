function fn_MFMC_add_user_dataset(MFMC, ref_or_index_or_loc, dataset_name, dataset_value)
%SUMMMAY
%   Adds user-specified dataset at specified location.
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   ref_or_index_or_loc - HDF5 reference, index or location of group to 
%   add to.
%   dataset_name - name of dataset
%   dataset_value - value of dataset
%--------------------------------------------------------------------------

dummy.(dataset_name)=dataset_value;

dataset_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname), '/', dataset_name];
%fn_hdf5_create_dataset(MFMC.fname, dataset_path, dataset_value);
fn_hdf5_add_to_dataset(dummy, MFMC.fname, [dataset_path],             'M', [], [size(dataset_value), inf], ndims(dataset_value)+1);

end