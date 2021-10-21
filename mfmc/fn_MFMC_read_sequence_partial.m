function SEQUENCE = fn_MFMC_read_sequence_partial(MFMC, ref_or_index_or_loc, datasets_to_include)
%SUMMMARY
%   Reads sequence data from file based on HDF5 reference, index or 
%   location. Datasets specified in sequence group are returned.
%   No checks are performed on data, other than 
%   checking that TYPE == SEQUENCE
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   ref_or_index_or_loc - HDF5 reference, index or location (relative to
%   MFMC.root_path) of sequence to read
%OUTPUTS
%   sequence - structured variable containing same fields as found in file. 
%--------------------------------------------------------------------------

sequence_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname, [MFMC.root_path, MFMC.sequence_name_template]), '/'];

[SEQUENCE, groups] = fn_hdf5_read_to_matlab(MFMC.fname, sequence_path, datasets_to_include, true);

if ~strcmp(SEQUENCE.TYPE, 'SEQUENCE')
    error('Data does not have TYPE = SEQUENCE');
end

end