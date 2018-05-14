function [time_data, error, LLC] = fn_ag_do_encoded_scan(FLR, direction, scan_steps, time_signals_per_step, varargin)

if nargin > 4
    time_out = varargin{1};
else
    time_out = 100;
end
if nargin > 5
    echo_on = varargin{2};
else
    echo_on = 0;
end
axis_no = 1;
%set initial position to zero
fn_ag_send_command('LCP 1 0', 0, echo_on);
fn_ag_send_command('FLM 3', 0 ,0); %perform CALS 0 on every inspection point
[raw_data, error]=fn_ag_send_command(sprintf('FLR %i %i %i', axis_no, FLR, direction), time_out, echo_on);

scan_header_length = 5;
signal_header_length = 8;
signal_header = raw_data(6:13);
bytes_per_signal = double(signal_header(2)) + double(signal_header(3)) * 2^8 + double(signal_header(4)) * 2^16;
dof = signal_header(7);
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
points_per_signal = (bytes_per_signal - signal_header_length) / bytes_per_point;
LLC = raw_data(end-17:end);

%check size of data
raw_data = raw_data(1:end - 18);
if rem(length(raw_data)-scan_header_length*scan_steps, bytes_per_signal) ~= 0
%     error('Incomplete data');
    disp('Incomplete data');
    return;
end;

%parse data
bytes_per_step = length(raw_data) / scan_steps;
time_data = reshape(raw_data, bytes_per_step, scan_steps);
axis_locations = time_data(1:5,:);
time_data = time_data(scan_header_length+1:end,:);
% time_signals_per_step = length(time_data(:,1))/bytes_per_signal;
time_data = reshape(time_data,bytes_per_signal,time_signals_per_step, scan_steps);
time_data = time_data(signal_header_length+1:end,:,:);

if bytes_per_point == 1
    time_data = double(time_data);
else
    time_data = double(time_data(2:2:end, :, :)) * 256 + double(time_data(1:2:end, :, :));
end;
time_data = 2 * time_data / fsd - 1;
return;

