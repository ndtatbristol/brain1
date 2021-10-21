function fn_hdf5_create_dataset(fname, dataset, data, varargin)
%general function for writing dataset, including size options

%If type not specified, class(data) is used
if length(varargin) < 1 || isempty(varargin{1})
    type = class(data);
else
    type = varargin{1};
end

%If max_size not specified, size(data) is used with singleton dims removed
if length(varargin) < 2 || isempty(varargin{2})
    max_size = size(data);
    if numel(max_size) == 2 
        max_size(max_size == 1) = []; 
    end
else
    max_size = varargin{2};
end

%If expandable_dimension not specified, size is assumed fixed
if length(varargin) < 3 || isempty(varargin{3})
    expandable_dimension = [];
else
    expandable_dimension = varargin{3};
end

%If deflate_value specified
if length(varargin) < 4 || isempty(varargin{4})
    deflate_value = [];
else
    deflate_value = varargin{4};
end

%set size of any expandable dimension to inf
for jj = 1:length(expandable_dimension)
    max_size(expandable_dimension(jj)) = inf;
end

%if any expandable dimensions, set chunk size, otherwise set to blank
if any(isinf(max_size))
    chunk_size = max_size;
    chunk_size(find(isinf(max_size))) = 1;
else
    chunk_size = [];
end

%create dataset
if strcmp(type, 'hdf5_ref') %special case - must be 8 cols
    if size(data, 2) ~= 8
        error('HDF5 reference data must have 8 columns')
    end
    h5_dims = size(data, 1);
    h5_maxdims = h5_dims;
    file_id = H5F.open(fname, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    space_id = H5S.create_simple(length(h5_dims), h5_dims, h5_maxdims);
    dset_id = H5D.create(file_id, dataset, 'H5T_STD_REF_OBJ', space_id, 'H5P_DEFAULT');
    H5D.write(dset_id,'H5ML_DEFAULT','H5S_ALL','H5S_ALL','H5P_DEFAULT', data');
    H5D.close(dset_id);
    H5S.close(space_id);
    H5F.close(file_id);
    return
    
else
    if ~isempty(chunk_size)
        if ~isempty(deflate_value)
            h5create(fname, dataset, max_size, 'Datatype', type, 'ChunkSize', chunk_size, 'Deflate', deflate_value);
        else
            h5create(fname, dataset, max_size, 'Datatype', type, 'ChunkSize', chunk_size);
        end
    else
        h5create(fname, dataset, max_size, 'Datatype', type);
    end
end

%write the data (unless data is empty)
if ~isempty(data)
    start = ones(size(max_size));
    count = max_size;
    jj = find(isinf(max_size));
    for ii = 1:length(jj)
        count(jj(ii)) = size(data, jj(ii));
    end
    h5write(fname, dataset, data, start, count);
end

end

