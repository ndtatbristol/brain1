function res = fn_ag_reset_tcpip(varargin);
timeout = 10;
reset_pause = 3;
if nargin < 1
    echo_on = 0;
else
    echo_on = varargin{1};
end;
if echo_on
    disp('fn_reset');
end;
[res, err] = fn_ag_send_command_tcpip('RST', timeout, echo_on);
pause(reset_pause);
return