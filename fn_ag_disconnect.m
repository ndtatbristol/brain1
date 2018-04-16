function res = fn_ag_disconnect(varargin)
%USAGE
%   res = fn_ag_disconnect(echo_on)
%INPUTS
%   echo_on - echos information to screen
%OUTPUTS
%   res - successful (1) or unsuccessful (0)
%NOTES
%   This function does nothing and is only included for completeness!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 1
    echo_on = 0;
else
    echo_on = varargin{1};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if echo_on
    disp('fn_ag_disconnect');
end
res = 1;
end