function time_data = fn_ag_do_test_tcpip(varargin)

global tcpip_obj

timeout = 5;
if nargin < 1
    echo_on = 0;
else
    echo_on = varargin{1};
end;
header_length = 8;
%call the do test command

[raw_result, err_msg] = fn_ag_send_command_tcpip('CALS 0', timeout, echo_on);

offset = 0;
if sum(raw_result(1:2))==3
   offset = 2;
end
%figure out how many points and bytes per point in first A-scan
bytes_per_signal = double(raw_result(2+offset)) + double(raw_result(3+offset)) * 2^8 + double(raw_result(4+offset)) * 2^16;
%bytes_per_signal = double(raw_result(2)) + double(raw_result(3)) * 2^8 + double(raw_result(4)) * 2^16;
dof = raw_result(7+offset);
switch dof
    case 1
        bytes_per_point = 1;
        sample_bits = 8;
    case 2
        bytes_per_point = 2;
        sample_bits = 10;
    case 3
        bytes_per_point = 2;
        sample_bits = 12;
    case 4
        bytes_per_point = 2;
        sample_bits = 16;
end;
fsd = 2 ^ sample_bits;
points_per_signal = (bytes_per_signal - header_length) / bytes_per_point;

%check size of data
raw_result = raw_result(1+offset:end - 2);
%keyboard
if rem(length(raw_result), bytes_per_signal) ~= 0
    error('Incomplete data');
    return;
end;

%parse data
no_time_traces = length(raw_result) / bytes_per_signal;
time_data = reshape(raw_result, bytes_per_signal, no_time_traces);
time_data = time_data(header_length+1:end,:);
if bytes_per_point == 1
    time_data = double(time_data);
else
    time_data = double(time_data(2:2:end, :)) * 256 + double(time_data(1:2:end, :));
end;
time_data = 2 * time_data / fsd - 1;
return;

