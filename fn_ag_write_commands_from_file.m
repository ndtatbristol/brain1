function fn_ag_write_commands_from_file(fname, varargin)
if nargin < 2
    echo_on = 0;
else
    echo_on = varargin{1};
end;
fid = fopen(fname, 'rt');
while ~feof(fid)
    str = fgetl(fid);
    fn_ag_send_command(str, 0, echo_on);
end;
fclose(fid);
return;