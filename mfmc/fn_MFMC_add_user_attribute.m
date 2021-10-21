function fn_MFMC_add_user_attribute(MFMC, ref_or_index_or_loc, attribute_name, attribute_value)
%SUMMMAY
%   Adds user-specified attribute at specified location.
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   ref_or_index_or_loc - HDF5 reference, index or location of group to 
%   add to.
%   attribute_name - name of attribute
%   attribute_value - value of attribute
%--------------------------------------------------------------------------
attribute_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname), '/', attribute_name];
fn_hdf5_create_attribute(MFMC.fname, attribute_path, attribute_value);
end