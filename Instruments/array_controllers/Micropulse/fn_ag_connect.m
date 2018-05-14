function res = fn_ag_connect(varargin)
%USAGE
%   res = fn_ag_connect(ip_address, port_no, echo_on)
%INPUTS
%   ip_address - IP address of device ['10.1.1.2']
%   port_no - port number [1067]
%   echo_on - echos information to screen
%OUTPUTS
%   res - successful (1) or unsuccessful (0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%default values
default_ip_address = '10.1.1.2';
default_port_no = 1067;
timeout = 10;
if nargin < 3
    echo_on = 0;
else
    echo_on = varargin{3};
end;
if nargin < 2
    port_no = default_port_no;
else
    port_no =  varargin{2};
end;
if nargin < 1
    ip_address = default_ip_address;
else
    ip_address = varargin{1};
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if echo_on
    disp('fn_ag_connect');
end;
ip_address_code = fn_addr_inet(ip_address);
[raw_result, err_msg] = fn_ag_send_command(sprintf('CONNECT %i %i', ip_address_code, port_no), timeout, 1);%for some reason connect does not work without echo on!
if strcmp(err_msg, 'Error no response')
    res = 0;
else
    res = 1;
end;
return;