function no_frames = fn_MFMC_get_no_frames(MFMC, ref_or_index_or_loc)
%SUMMMAY
%   Returns number of frames in specified sequence
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   ref_or_index_or_loc - HDF5 reference, index or location of sequence
%OUTPUTS
%   no_frames - number of frames in sequence
%--------------------------------------------------------------------------

dataspace_dim = fn_MFMC_get_data_dimensions(MFMC, ref_or_index_or_loc,'MFMC_DATA');

if (isempty(dataspace_dim))
    no_frames = 0;
else
    no_frames = dataspace_dim(3);
end


end
