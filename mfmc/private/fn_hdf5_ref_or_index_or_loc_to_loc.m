function loc = fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, fname, template_loc)
%Checks ref_or_index_or_loc and converts to hdf5 location
if strcmp(class(ref_or_index_or_loc), 'uint8') && all(size(ref_or_index_or_loc) == [1, 8])
    %ref_or_index_or_name is ref
%     loc = fn_hdf5_ref_to_location(fname, ref_or_index_or_loc);
    file_id = H5F.open(fname);
    loc = H5R.get_name(file_id, 'H5R_OBJECT', ref_or_index_or_loc);
    H5F.close(file_id);
elseif isnumeric(ref_or_index_or_loc)
    %ref_or_index_or_name is index
    loc = sprintf(template_loc, ref_or_index_or_loc); 
elseif ischar(ref_or_index_or_loc)
    %ref_or_index_or_name is string
    loc = ref_or_index_or_loc; 
else
    error('Invalid reference');
end

try 
    h5info(fname, loc);
catch
    error('No matching object in file');
end

end
