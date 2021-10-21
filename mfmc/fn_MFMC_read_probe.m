function PROBE = fn_MFMC_read_probe(MFMC, ref_or_index_or_loc)
%SUMMMAY
%   Reads probe data from file. All datasets in probe group are returned.
%   No checks are performed on data, other than checking that TYPE == PROBE
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   ref_or_index_or_loc - HDF5 reference, index or location (relative to
%   MFMC.root_path) of probe to read
%OUTPUTS
%   PROBE - structured variable containing same fields as found in file. 
%--------------------------------------------------------------------------

probe_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname, [MFMC.root_path, MFMC.probe_name_template]), '/'];

PROBE = fn_hdf5_read_to_matlab(MFMC.fname, probe_path);

if ~strcmp(PROBE.TYPE, 'PROBE')
    error('Data does not have TYPE = PROBE');
end

end

