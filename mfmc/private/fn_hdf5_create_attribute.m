function fn_hdf5_create_attribute(fname, attribute, data)
%general function for writing attribute to any location regardless of
%whether it already exists (so equivalent to high level h5create function
%for datasets

%make sure group exists
[groups, name] = fn_hdf5_decompose_location(attribute);
fn_hdf5_create_group(fname, groups{end});

h5writeatt(fname, groups{end}, name, data);

end

