function dataspace_dim = fn_MFMC_get_data_dimensions(MFMC, ref_or_index_or_loc,dataspace_name, varargin)
%SUMMMAY
%   Returns dataspace dimensions for specified dataset within specified reference, index or location
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   ref_or_index_or_loc - HDF5 reference, index or location
%   dataspace_name - dataspace name, e.g. MFMC_DATA within a SEQUENCE
%   (optional) object_type - string 'PROBE','LAW' or 'SEQUENCE', used if
%   index specified in ref_or_index_or_loc, if not present then assumed to
%   be 'SEQUENCE'
%OUTPUTS
%   dataspace_dim - array of each dimension of specified dataset if it exists
%--------------------------------------------------------------------------

%check ref_or_index_or_loc is numeric, i.e. index (but not ref)
if (isnumeric(ref_or_index_or_loc) && ~(isa(ref_or_index_or_loc, 'uint8') && all(size(ref_or_index_or_loc) == [1, 8])))
    % use index together with object_type if available to get path
    if (~isempty(varargin))
        object_type=varargin{1};
    else
        object_type='SEQUENCE';
    end
    switch object_type
        case 'PROBE'
            object_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname, [MFMC.root_path, MFMC.probe_name_template]), '/'];
        case 'LAW'
            object_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname, [MFMC.root_path, MFMC.law_name_template]), '/'];
        otherwise
            object_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname, [MFMC.root_path, MFMC.sequence_name_template]), '/'];
    end
else
    % use ref or location directly to get path
    object_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname, [MFMC.root_path, MFMC.probe_name_template]), '/'];
end

% get information from within HDF5 file on specific dataspace if it exists
try
    dataspace_dim = h5info(MFMC.fname, [object_path, dataspace_name]);
    dataspace_dim = dataspace_dim.Dataspace.Size;
catch
    %case if doesn't exist
    dataspace_dim=[];
end
        
end
