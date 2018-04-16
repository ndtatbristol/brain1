function res = fn_ag_connect_tcpip(varargin)
%USAGE
%   res = fn_ag_connect(ip_address, port_no, echo_on)
%INPUTS
%   ip_address - IP address of device ['10.1.1.2']
%   port_no - port number [1067]
%   echo_on - echos information to screen
%OUTPUTS
%   res - successful (1) or unsuccessful (0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global tcpip_obj



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
    disp('fn_ag_connect_tcpip');
end;

tcpip_obj=tcpip(ip_address,port_no);
set(tcpip_obj,'ByteOrder','littleEndian');
set(tcpip_obj,'Terminator','CR');
set(tcpip_obj,'InputBufferSize',32);
set(tcpip_obj, 'BytesAvailableFcnCount', tcpip_obj.InputBufferSize);
set(tcpip_obj,'ReadAsyncMode', 'continuous');
set(tcpip_obj,'BytesAvailableFcnMode', 'byte');
set(tcpip_obj,'BytesAvailableFcn', @fn_callback_tcpip);
fopen(tcpip_obj)

res=1;
end

