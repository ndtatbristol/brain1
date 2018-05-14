function res = fn_DSL_connect(varargin)
%USAGE
%   res = fn_ag_connect(echo_on)
%INPUTS
%   echo_on - echos information to screen
%OUTPUTS
%   res - successful (1) or unsuccessful (0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%default values
if nargin < 1
    echo_on = 0;
else
    echo_on = varargin{1};
end;

%first load the DSL library
if ~libisloaded('DSLFITacquire')
    disp('Loading Library');
    loadlibrary('DSLFITacquire');
end;

%launch DSL FIT Scan with default options
res = calllib('DSLFITacquire','LaunchDSLFITscan','DSLFITacquire.vi','','C:\Users\Public\Documents\FIToolbox\Configs\System\FITsystem.cfg','C:\Users\Public\Documents\FIToolbox\Configs\Setups\Default.cfg');

if res == 0
    res = 1;
else
    res = 0;
end;

%return
return;