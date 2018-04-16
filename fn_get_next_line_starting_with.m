function [whole_line, rest_of_line, end_of_file] = fn_get_next_line_starting_with(str, fid)
whole_line = '';
end_of_file = 0;
while ~strncmp(whole_line, str, length(str)) & ~end_of_file
    end_of_file = feof(fid);
    if ~end_of_file
        whole_line = strtrim(fgetl(fid));
    end;
end;
if end_of_file
    whole_line = '';
    rest_of_line = '';
    return;
end;
rest_of_line = strtrim(whole_line(length(str) + 1:end));
return