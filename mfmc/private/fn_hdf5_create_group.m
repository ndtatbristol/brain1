function group_ref = fn_hdf5_create_group(fname, location)

%creates group at specified location (i.e. of location ='/a/b/c' then the
%group created is c), and creates intervening groups if required.

%open file
try
    file_id = H5F.open(fname,'H5F_ACC_RDWR','H5P_DEFAULT');
catch
    error(['Failed to open file: ', fname]);
end

if length(location) > 1 && location(end) == '/'
    location = location(1:end-1);
end

[groups, name] = fn_hdf5_decompose_location(location);

%if not already a group create it and intervening groups
if ~strcmp(fn_hdf5_get_object_type(fname, location), 'group')
    %work recursively, creating any missing groups on the way like h5write does
    for ii = 2:length(groups)
        if strcmp(fn_hdf5_get_object_type(fname, groups{ii}), 'does not exist')
            [~, intervening_name] = fn_hdf5_decompose_location(groups{ii});
            fn_create_group(file_id, groups{ii-1}, intervening_name)
        end
    end
    fn_create_group(file_id, groups{end}, name);
end
H5F.close(file_id);

group_ref = fn_hdf5_get_ref(fname, location);
end

function fn_create_group(file_id, location, name)
location_id = H5G.open(file_id, location);
group_id = H5G.create(location_id, name, 'H5P_DEFAULT', 'H5P_DEFAULT', 'H5P_DEFAULT');
H5G.close(group_id);
H5G.close(location_id);
end





