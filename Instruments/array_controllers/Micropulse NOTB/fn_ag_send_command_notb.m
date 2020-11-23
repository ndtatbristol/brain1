function [res, err]=fn_ag_send_command_notb(cmd_str,timeout,echo,varargin);

if nargin==3
    buffersize0=32;
else
    buffersize0=varargin{1};
end

global tcpip_obj

if echo
    disp(cmd_str)
end


%cmd_str = cast(cmd_str, 'int8');
cmd_str=uint8(cmd_str);
cmd_str=[cmd_str 13]; %13 on the end adds a carriage return 
%cmd_str
if timeout==0
    write(tcpip_obj,cmd_str)
else
    %readasync(tcpip_obj);
    
    write(tcpip_obj,cmd_str);
    
    while tcpip_obj.BytesAvailable < buffersize0
        %drawnow
        pause(0.01)
    end
    res = read(tcpip_obj)';
    %stopasync(tcpip_obj)
end
tmp=read(tcpip_obj);
tmp=[];
err=[];
end

