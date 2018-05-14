function res = fn_DSL_do_test(varargin)
%USAGE
%   res = fn_DSL_do_test(echo_on)
%INPUTS
%   filename
%   echo_on - echos information to screen
%OUTPUTS
%   res - successful (1) or unsuccessful (0)
%NOTES
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%default values
if nargin < 7
    res = 0;
    disp('Not all arguements present');
    return
else
    total_points = varargin{1};
    num_ascans = varargin{2};
    length_ascan = varargin{3};
    tx_no = varargin{4};
    rx_no = varargin{5};
    is_hmc = varargin{6};
    echo_on = varargin{7};
end;


% Get pointer to where we will save data
Stream = zeros(1,total_points);
i16Stream = int16(Stream);
clearvars Stream;
pi16Stream = libpointer('int16Ptr', i16Stream);
NumFrames = int32(is_hmc);
NumTx = int32(tx_no);
NumRx = int32(rx_no);
NumSamp = int32(length_ascan);
pNumFrames = libpointer('int32Ptr', NumFrames);
pNumTx = libpointer('int32Ptr', NumTx);
pNumRx = libpointer('int32Ptr', NumRx);
pNumSamp = libpointer('int32Ptr', NumSamp);

%get latest frame from the device
% int32_t __stdcall GetCustomAscans(int32_t NumAscans, int32_t Frames[0], 
% 	int32_t Tx[], int32_t Rx[], int32_t Samples[], int32_t timeoutms,
% 	int16_t Ascans[], int32_t AscansLength);
res = calllib('DSLFITacquire','GetCustomAscans', num_ascans, pNumFrames, pNumTx, pNumRx, pNumSamp, 10000, pi16Stream, total_points);
res = reshape(pi16Stream.value, length_ascan, num_ascans);

return;

