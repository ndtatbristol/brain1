function [group_name, group_index] = fn_hdf5_next_group_name(fname, location, group_name_template)
%SUMMARY
%   Returns next unused group name in fname that matches [location,
%   group_name_template], incrementing index in group_name_template.
a = h5info(fname, location);
if ~isempty(a.Groups)
    current_group_names = {a.Groups.Name}';
    group_index = 0;
    for ii = 1:length(a.Groups)
        tmp = sscanf(current_group_names{ii}, [location, group_name_template]);
        if ~isempty(tmp)
            group_index = max(group_index, tmp);
        end
    end
    group_index = group_index + 1;
else
    group_index = 1;
end
group_name = sprintf(group_name_template, group_index);
end