function ref = fn_hdf5_get_ref(fname, object)
%returns reference to HDF5 object (group or dataset, not attribute)

[groups, name] = fn_hdf5_decompose_location(object);

%deal with root issue
if isempty(name) 
    ref = [];
    return
end

%open file
try
    file_id = H5F.open(fname,'H5F_ACC_RDWR','H5P_DEFAULT');
catch
    error(['Failed to open file: ', fname]);
end
location_id = H5G.open(file_id, groups{end});
ref = H5R.create(location_id, name, 'H5R_OBJECT', -1);
H5G.close(location_id);
H5F.close(file_id);
end