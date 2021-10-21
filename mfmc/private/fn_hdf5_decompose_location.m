function [groups, name] = fn_hdf5_decompose_location(location)
%Splits single hdf5 object location string up into list of groups and name
%of object

jj = findstr(location, '/');

if isempty(jj) || jj(1) ~= 1
    error('Invalid hdf5 location');
end
    
groups{1} = '/';
for ii = 2:length(jj)
    groups{ii} = location(1:jj(ii) - 1);
end
name = location(jj(end)+1:end);
end