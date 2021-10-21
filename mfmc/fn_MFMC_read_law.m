function law = fn_MFMC_read_law(MFMC, ref)
%SUMMMARY
%   Reads focal law data from file based on HDF5 reference. No checks are 
%   performed on data, other than checking that TYPE == LAW
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   ref - HDF5 reference of law to read
%OUTPUTS
%   law - structured variable containing same fields as found in file. 
%--------------------------------------------------------------------------

if strcmp(class(ref), 'uint8') && all(size(ref) == [1, 8])
    law_path = fn_hdf5_ref_or_index_or_loc_to_loc(ref, MFMC.fname, []);
else
    error('Laws must be accessed by HDF5 reference');
end

law = fn_hdf5_read_to_matlab(MFMC.fname, law_path);

if ~strcmp(law.TYPE, 'LAW')
    error('Data does not have TYPE = LAW');
end

end

