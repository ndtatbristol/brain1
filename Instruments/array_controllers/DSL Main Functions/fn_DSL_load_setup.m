function res = fn_DSL_load_setup(varargin)
%USAGE
%   res = fn_DSL_load_setup(varargin)
%INPUTS
%   setupfilename
%   echo_on - echos information to screen
%OUTPUTS
%   res - successful (1) or unsuccessful (0)
%NOTES
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%default values
if nargin < 2
    res = 0;
    disp('Not all arguements present');
    return
else
    setupfilename = varargin{1};
    echo_on = varargin{2};
end;

%Set values
ResponseFilePath = setupfilename;
ResponseMessage = setupfilename;

pResponseFilePath = libpointer('voidPtr', [int8(ResponseFilePath) 0]);
pResponseMessage = libpointer('voidPtr', [int8(ResponseMessage) 0]);

%Load setup file into DSLFITacquire
res = calllib('DSLFITacquire','LoadSaveFile',1,setupfilename,10000, pResponseFilePath, 16, pResponseMessage, 16);

%return
return;