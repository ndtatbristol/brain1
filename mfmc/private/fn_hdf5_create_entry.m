function fn_hdf5_create_entry(data, fname, location, M_or_O, A_or_D, varargin)
%USAGE
%   fn_hdf5_create_entry(data, fname, location, M_or_O, A_or_D [, type, max_size, expandable_dimension])
%SUMMARY
%   General function for trying write field from Matlab structure to hdf5 
%   file.
%INPUTS
%OUTPUTS

%--------------------------------------------------------------------------

[groups, name] = fn_hdf5_decompose_location(location);

%if optional and field does not exist, do nothing return 
if ~isempty(data) && ~isfield(data, name) && strcmp(M_or_O, 'O')
    return
end

if isempty(data)
    obj = [];
else
    obj = data.(name);
end

%If type not specified, class(data.(name)) is used
if length(varargin) < 1 || isempty(varargin{1})
    type = class(obj);
else
    type = varargin{1};
end

%If max_size not specified, size(data.(name)) is used with singletons
%removed
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

switch A_or_D
    case 'A'
        fn_hdf5_create_attribute(fname, location, obj);
    case 'D'
        fn_hdf5_create_dataset(fname, location, obj, type, max_size, expandable_dimension);
end

end
