function fn_hdf5_add_to_dataset(data, fname, location, M_or_O, varargin)
%USAGE
%   fn_hdf5_add_to_dataset(data, fname, location [, type, max_size, expandable_dimension])
%SUMMARY
%   General function for adding field from Matlab structure to expandable hdf5 
%   dataset.
%INPUTS
%OUTPUTS

%--------------------------------------------------------------------------

[groups, name] = fn_hdf5_decompose_location(location);

if isfield(data, name) || strcmp(M_or_O, 'M')
    obj = data.(name);
else
    return
end

%If type not specified, class(data.(name)) is used
if length(varargin) < 1 || isempty(varargin{1})
    type = class(obj);
else
    type = varargin{1};
end

%If max_size not specified, size(data.(name)) is used
if length(varargin) < 2 || isempty(varargin{2})
    max_size = size(obj);
    if numel(max_size) == 2 
        max_size(max_size == 1) = []; 
    end
else
    max_size = varargin{2};
end

%If expandable_dimension not specified, size is assumed fixed (only relevant for
%datasets)
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

try
    tmp = h5info(fname, location);
    count = ones(size(tmp.Dataspace.Size));
    count(1:numel(size(obj))) = size(obj);
    start = ones(size(tmp.Dataspace.Size));
    for jj = 1:length(  expandable_dimension)
        start(expandable_dimension(jj)) = tmp.Dataspace.Size(expandable_dimension(jj)) + 1;
    end
    h5write(fname, location, obj, start, count);
catch
    %create if not present%%%this should be better to make sure other
    %errors are properly caught
    fn_hdf5_create_dataset(fname, location, obj, type, max_size, expandable_dimension, deflate_value);
end


end
