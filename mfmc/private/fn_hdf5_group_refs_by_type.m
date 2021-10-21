function [OBJ] = fn_hdf5_group_refs_by_type(fname, location, type)
%SUMMARY
%   Returns HDF5 object references to all groups at location with attribute
%   TYPE = type
a = h5info(fname, location);
OBJ = [];
if ~isempty(a.Groups)
    for ii = 1:length(a.Groups)
        b = h5info(fname, a.Groups(ii).Name);
        jj = find(strcmp({b.Attributes(:).Name}, 'TYPE'));
        if strcmp(b.Attributes(jj).Value, type)
            kk = length(OBJ) + 1;
            OBJ{kk}.ref = fn_hdf5_get_ref(fname, a.Groups(ii).Name);
            OBJ{kk}.location = a.Groups(ii).Name;
            OBJ{kk}.name = OBJ{kk}.location(length(location) + 1: end);
        end
    end
end

end