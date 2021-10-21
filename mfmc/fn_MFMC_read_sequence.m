function SEQUENCE = fn_MFMC_read_sequence(MFMC, ref_or_index_or_loc)
%SUMMMARY
%   Reads sequence data from file based on HDF5 reference, index or 
%   location. All datasets in sequence group are returned, except for the 
%   actual MFMC data itself. No checks are performed on data, other than 
%   checking that TYPE == SEQUENCE
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   ref_or_index_or_loc - HDF5 reference, index or location (relative to
%   MFMC.root_path) of sequence to read
%OUTPUTS
%   sequence - structured variable containing same fields as found in file. 
%--------------------------------------------------------------------------

sequence_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname, [MFMC.root_path, MFMC.sequence_name_template]), '/'];

datasets_to_exclude = {'MFMC_DATA', 'MFMC_DATA_IM'};

[SEQUENCE, groups] = fn_hdf5_read_to_matlab(MFMC.fname, sequence_path, datasets_to_exclude);

if ~strcmp(SEQUENCE.TYPE, 'SEQUENCE')
    error('Data does not have TYPE = SEQUENCE');
end

end