function [res, err]=fn_ag_send_command_tcpip(cmd_str,timeout,echo);

global tcpip_obj
global data_call

data_call=0;

if echo
    disp(cmd_str)
end

flushinput(tcpip_obj)
res=[];

if timeout==0
    fprintf(tcpip_obj,cmd_str)
else
    readasync(tcpip_obj);
    
    fprintf(tcpip_obj,cmd_str);
    
    while data_call ==0
        drawnow
    end
    res = fread(tcpip_obj,tcpip_obj.BytesAvailable,'uint8')';
    stopasync(tcpip_obj)
end
err=[];
end

