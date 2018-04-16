function [test_options, system_info] = fn_ag_get_test_options_tcpip(varargin)
timeout = 10;
if nargin < 1
    echo_on = 0;
else
    echo_on = varargin{1};
end;
test_options = [];
system_info = [];
%request reset parameters without reset to get sample frequency and system
%info as bonus
[raw_result, err_msg] = fn_ag_send_command_tcpip('STS -1', timeout, echo_on);
%check header byte
if raw_result(1) ~= hex2dec('23')
    return;
end;
dof_mode = raw_result(8);
switch dof_mode
    case 1
        test_options.sample_bits = 8;
    case 2
        test_options.sample_bits = 10;
    case 3
        test_options.sample_bits = 12;
    case 4
        test_options.sample_bits = 16;
end;
test_options.sample_freq = double(raw_result(10)) * 1e6; %this doesn't work but should!
test_options.dof_mode = raw_result(8);
system_info.max_channels = raw_result(3);
system_info.hardware_version = raw_result(6:7);
system_info.main_processor_sw_version = raw_result(13:16);
system_info.ethernet_processor_sw_version = raw_result(29:32);
system_info.rf_slot = raw_result(18:27);

[raw_result, err_msg] = fn_ag_send_command_tcpip('XXA 256', timeout, echo_on);
raw_result = cast(raw_result, 'uint16');
if raw_result(1) ~= hex2dec('20')
    return;
end;
% test_options.db_gain = bin2dec([dec2bin(raw_result(7)), dec2bin(raw_result(6))])/4;
test_options.db_gain = (bitshift(raw_result(7), 8) + raw_result(6)) / 4;
test_options.filter_no = bitand(raw_result(8), bin2dec('00001111'));
test_options.smoothing = bitshift(bitand(raw_result(8), bin2dec('11110000')), -4);
gate_start = bitshift(raw_result(14), 8) + raw_result(13);
gate_end = bitshift(raw_result(16), 8) + raw_result(15);
test_options.time_pts = gate_end - gate_start;
test_options.gate_start = double(gate_start) / test_options.sample_freq;
return;

