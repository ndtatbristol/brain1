function res = fn_DSL_disconnect(varargin)
%USAGE
%   res = fn_DSL_disconnect(echo_on)
%INPUTS
%   echo_on - echos information to screen
%OUTPUTS
%   res - successful (1) or unsuccessful (0)
%NOTES
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 1
    echo_on = 0;
else
    echo_on = varargin{1};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if echo_on
    disp('fn_DSL_disconnect');
end

if libisloaded('DSLFITacquire') ~= 1 %catch the initial call to disconnect when BRAIN is loaded
    res = 1;
    return;
end

%Set values
ResponseMessage = 'This is the array the response will be written into';

pResponseMessage = libpointer('voidPtr', [int8(ResponseMessage) 0]);

%shutdown the aquisition program
res = calllib('DSLFITacquire','SetState','Shutdown', 10000, pResponseMessage, 16);

if res == 0
    unloadlibrary('DSLFITacquire');
    res = 1;
else
    disp('Unable to shut down acquisition program');
    res = 0;
end;

%return
return;