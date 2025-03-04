function [tx_no, rx_no] = fn_ag_set_test_options_tcpip(test_options, varargin);
default_test_options.sample_freq = 25e6;
default_test_options.pulse_voltage = 100;
default_test_options.pulse_width = 80e-9;
default_test_options.time_pts = 1000;
default_test_options.sample_bits = 16;
default_test_options.db_gain = 30;
default_test_options.tx_ch = 1;
default_test_options.rx_ch = 1;
default_test_options.filter_no = 4;
default_test_options.smoothing = 1;
default_test_options.prf = 5000;
default_test_options.averages = 1;
default_test_options.gate_start = 0;
default_test_options.header_length = 8;
default_test_options.buff_mode = 0;
default_test_options.bytes_per_point = 2;

global tcpip_obj
fclose(tcpip_obj)
set(tcpip_obj,'InputBufferSize',32);
set(tcpip_obj, 'BytesAvailableFcnCount', tcpip_obj.InputBufferSize);
fopen(tcpip_obj);


%set default options
if nargin < 2
    echo_on = 0;
else
    echo_on = varargin{1};
end;
test_options = fn_set_default_fields(test_options, default_test_options);

if ~isfield(test_options,'tx_delay_law_array')
    test_options.tx_delay_law_array=zeros(size(test_options.tx_ch));
end

if ~isfield(test_options,'rx_delay_law_array')
    test_options.rx_delay_law_array=zeros(size(test_options.tx_ch));
end

%clear focal laws
fn_ag_send_command_tcpip('DXN 0 0', 0 , echo_on);
fn_ag_send_command_tcpip('DLYS 0 0', 0, echo_on);


fn_ag_set_sample_freq(test_options.sample_freq, echo_on);
fn_ag_set_sample_bits(test_options.sample_bits, echo_on);

%PRF
fn_ag_set_prf(test_options.prf, echo_on);

%define the FMC
[tx_no, rx_no] = fn_ag_define_fmc(test_options.tx_ch, test_options.rx_ch, test_options.tx_delay_law_array, test_options.rx_delay_law_array, echo_on);

%set gain - must be called after fn_define_fmc
fn_ag_set_db_gain(test_options.db_gain, echo_on);

%time points
fn_ag_set_time_pts(test_options.time_pts, test_options.gate_start, 1/test_options.sample_freq, echo_on);

%pulse details
fn_ag_set_pulse_voltage(test_options.pulse_voltage, echo_on);
fn_ag_set_pulse_width(test_options.pulse_width, echo_on);

%filter
fn_ag_set_filter(test_options.filter_no, test_options.smoothing, echo_on);

%set full waveform params
fn_ag_send_command_tcpip('AWF 0 1', 0, echo_on);%specify full RF signal
fn_ag_send_command_tcpip('AWFS 0 1', 0, echo_on);%specify full RF signal
fn_ag_set_averages(test_options.averages, echo_on); %sets averages and also specifes A-scans returned

el_num=size(test_options.tx_ch,1);

if test_options.averages >1 | el_num > 128
    test_options.buff_mode = 1;
end

fn_ag_send_command_tcpip(sprintf('BUFF %i', test_options.buff_mode), 0, echo_on);

if test_options.sample_bits == 8
    test_options.bytes_per_point = 1;
end
    
buffer_size = (test_options.header_length+(test_options.time_pts.* test_options.bytes_per_point)) .* length(tx_no)  + 2 + 2.*test_options.buff_mode; %the plus to is to account for the [1 1] at the end of a message
%change the size of the input buffer to capture experimental data
fclose(tcpip_obj)
set(tcpip_obj,'InputBufferSize',buffer_size);
set(tcpip_obj, 'BytesAvailableFcnCount', tcpip_obj.InputBufferSize);
fopen(tcpip_obj);

end

%--------------------------------------------------------------------------

%averages
function actual_avs = fn_ag_set_averages(avs, echo_on)
pow2 = nextpow2(abs(avs));
pow2 = min([pow2, 6]);%don't know what max avs bvalue allowed is! to check
pow2 = max([pow2, 0]);
fn_ag_send_command_tcpip(sprintf('AMP 0 13 %i', pow2), 0, echo_on);%avs and FMC mode
fn_ag_send_command_tcpip(sprintf('AMPS 0 13 %i', pow2), 0, echo_on);%avs and FMC mode
actual_avs = 2 ^ pow2;
end

%PRF
function actual_prf = fn_ag_set_prf(prf, echo_on)
prf = round(prf / 10) * 10;
prf = min([prf, 20000]);
prf = max([prf, 1]);
actual_prf = prf;
fn_ag_send_command_tcpip(sprintf('PRF %i', actual_prf), 0, echo_on);%PRF
end

%sample frequency
function actual_sample_freq = fn_ag_set_sample_freq(sample_freq, echo_on)
reset_pause = 3;
[test_options, system_info] = fn_ag_get_test_options_tcpip(echo_on);
if sample_freq == test_options.sample_freq
    actual_sample_freq = sample_freq;
    return;
end;
sample_freq = interp1([10, 25, 50, 100] * 1e6, [10, 25, 50, 100] * 1e6, sample_freq, 'nearest', 'extrap');
actual_sample_freq = sample_freq;
[res, err] = fn_ag_send_command_tcpip(sprintf('RST %i', round(sample_freq / 1e6)), 10, echo_on);
pause(reset_pause);
end

%sample bits (DOF mode)
function actual_bits = fn_ag_set_sample_bits(bits, echo_on)
bits = round(bits / 2) * 2;
bits = min([bits, 16]);
bits = max([bits, 8]);
switch bits
    case 8
        dof_val = 1;
    case 10
        dof_val = 2;
    case 12
        dof_val = 3;
    otherwise
        dof_val = 4;
        bits = 16;
end;
fn_ag_send_command_tcpip(sprintf('DOF %i', dof_val), 0, echo_on);%pulse amp
actual_bits = bits;
end

%pulse voltage
function actual_voltage = fn_ag_set_pulse_voltage(voltage, echo_on)
volt1 = round(voltage / 5) * 5;
volt1 = min([volt1, 200]);
volt1 = max([volt1, 50]);
fn_ag_send_command_tcpip(sprintf('PAV 1 128 %i', volt1), 0, echo_on);%pulse amp
actual_voltage = volt1;
end

%pulse width
function actual_width = fn_ag_set_pulse_width(width, echo_on)
width_ns = round(width * 1e9 / 2) * 2;
width_ns = max([width_ns, 20]);
width_ns = min([width_ns, 500]);
actual_width = width_ns / 1e9;
fn_ag_send_command_tcpip(sprintf('PAW 1 128 %i', width_ns), 0, echo_on);%pulse width
end

%time points
function actual_pts = fn_ag_set_time_pts(pts, gate_start, time_step, echo_on)
pts = min([pts, 96000]);
pts = max([pts, 2]);
gate_start = round(gate_start/time_step);
fn_ag_send_command_tcpip(sprintf('GAT 0 %i %i', gate_start, gate_start+pts), 0, echo_on);%time points
fn_ag_send_command_tcpip(sprintf('GATS 0 %i %i', gate_start, gate_start+pts), 0, echo_on);%time points
actual_pts = pts;
end

%set up filter
function [actual_fmin, actual_fmax, actual_smoothing] = fn_ag_set_filter(filter_no, smoothing, echo_on)
filter_no = round(filter_no);
filter_no = min([filter_no, 4]);
filter_no = max([filter_no, 1]);
smoothing = max([smoothing, 1]);
smoothing = min([smoothing, 8]);
actual_smoothing = smoothing;
fn_ag_send_command_tcpip(sprintf('FRQS 0 %i %i', filter_no, smoothing), 0, echo_on);
%set filter (FRQS) %check effect of last param on RF data - should be none
switch filter_no
    case 1
        actual_fmin = 5e6;
        actual_fmax = 10e6;
    case 2
        actual_fmin = 2e6;
        actual_fmax = 10e6;
    case 3
        actual_fmin = 0.75e6;
        actual_fmax = 5e6;
    case 4
        actual_fmin = 0.75e6;
        actual_fmax = 20e6;
end;
end

%set up focal laws
function [tx_no, rx_no] = fn_ag_define_fmc(tx_ch, rx_ch, tx_delay_law_array, rx_delay_law_array, echo_on)

transmit_laws = size(tx_ch, 1);
time_traces = length(find(rx_ch));
tx_no=zeros(time_traces,1)';
rx_no=zeros(time_traces,1)';
use_els=size(tx_delay_law_array,2);
counter = 0;

tx_delay_law_array = round(tx_delay_law_array/1e-9);
rx_delay_law_array = round(rx_delay_law_array/1e-9);

for fl_ii = 1:transmit_laws %loop through focal laws

    %clear existing tx delays
    for tx_ii = 1:use_els %loop through focal laws
        fn_ag_send_command_tcpip(sprintf('TXF %i %i -1', fl_ii, tx_ii), 0, echo_on);%law, ch, del
    end;
    %find transmitters for each focal law (i.e each row of the tx or rx_matrix
    tx_nos = find(tx_ch(fl_ii,:));
    for tx_ii = 1:length(tx_nos) %add each transmitter specified for focal law
        fn_ag_send_command_tcpip(sprintf('TXF %i %i %i', fl_ii, tx_nos(tx_ii), tx_delay_law_array(fl_ii,tx_nos(tx_ii))), 0, echo_on);%law, ch, del
    end;
    %clear existing rx delays
    fn_ag_send_command_tcpip(sprintf('RXF %i 0 -1 0', fl_ii), 0, echo_on);%law, ch, del
    rx_nos = find(rx_ch(fl_ii,:));
    for rx_ii = 1:length(rx_nos); %add receivers to all focal laws
        counter = counter + 1;
        fn_ag_send_command_tcpip(sprintf('RXF %i %i %i 0', fl_ii, rx_nos(rx_ii), rx_delay_law_array(fl_ii,rx_nos(rx_ii))), 0, echo_on);%law, ch, del, trim_amp

        if length(tx_nos)>1
            tx_no(counter) = fl_ii;
        else
            tx_no(counter) = tx_nos;
        end

        rx_no(counter) = rx_nos(rx_ii);
    end;
    %assign focal laws to tests starting at 256
    fn_ag_send_command_tcpip(sprintf('TXN %i %i', 255 + fl_ii, fl_ii), 0, echo_on);
    fn_ag_send_command_tcpip(sprintf('RXN %i %i', 255 + fl_ii, fl_ii), 0, echo_on);
end;

%assign tests to sweep 1
if size(tx_ch, 1) > 1
    fn_ag_send_command_tcpip(sprintf('SWP 1 %i - %i',256 ,255 + size(tx_ch, 1)), 0, echo_on);
else
    fn_ag_send_command_tcpip(sprintf('SWP 1 %i',256), 0, echo_on);
end;
%disable all sweeps
fn_ag_send_command_tcpip('DIS 0', 0, echo_on);
fn_ag_send_command_tcpip('DISS 0', 0, echo_on);

%enable test 256
fn_ag_send_command_tcpip('ENA 256', 0, echo_on);
%enable sweep 1
fn_ag_send_command_tcpip('ENAS 1', 0, echo_on);
end

function actual_db_gain = fn_ag_set_db_gain(db_gain, echo_on)
db_gain = min([db_gain, 70]);
db_gain = max([db_gain, 0]);
gain_val = round(db_gain * 4);
fn_ag_send_command_tcpip(sprintf('GAN 0 %i', gain_val), 0, echo_on);%time points
fn_ag_send_command_tcpip(sprintf('GANS 0 %i', gain_val), 0, echo_on);%time points
actual_db_gain = gain_val / 4;
end