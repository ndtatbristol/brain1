function res = fn_ag_disconnect_tcpip(varargin)
%USAGE
%   res = fn_ag_disconnect(echo_on)
%INPUTS
%   echo_on - echos information to screen
%OUTPUTS
%   res - successful (1) or unsuccessful (0)
%NOTES
%   This function does nothing and is only included for completeness!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global tcpip_obj

if nargin < 1
    echo_on = 0;
else
    echo_on = varargin{1};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if echo_on
    disp('fn_ag_disconnect');
end

if exist('tcpip_obj') && isa(tcpip_obj, 'tcpip')
    try
        fclose(tcpip_obj);
        delete(tcpip_obj);
        clear('tcpip_obj');
    catch
        
    end
end

res = 1;
end