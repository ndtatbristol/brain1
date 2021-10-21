function object_type = fn_hdf5_get_object_type(fname, object_name)
object_type = 'does not exist';
try
    file_id = H5F.open(fname, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
catch
    error(['Failed to open file: ', fname]);
end
c1 = onCleanup(@()H5F.close(file_id));

try
    obj_id = H5O.open(file_id, object_name, 'H5P_DEFAULT');
    c2 = onCleanup(@()H5O.close(obj_id));
catch
    %could still be an attribute
    try 
        jj = findstr(object_name, '/');
        loc = object_name(1:jj(end)-1);
        attr_name = object_name(jj(end)+1:end);
        attval = h5readatt(fname, loc, attr_name);
        object_type = 'attribute';
        return
    catch
        return
    end
end

obj_info = H5O.get_info(obj_id);

switch(obj_info.type)
       case H5ML.get_constant_value('H5G_LINK')
           object_type = 'link';
       case H5ML.get_constant_value('H5G_GROUP')
           object_type = 'group';
       case H5ML.get_constant_value('H5G_DATASET')
           object_type = 'dataset';
       case H5ML.get_constant_value('H5G_TYPE')
           object_type = 'named datatype';


end