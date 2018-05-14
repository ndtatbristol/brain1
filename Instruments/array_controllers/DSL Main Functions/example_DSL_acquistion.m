%This example file connects to the DSL array controller, captures
%one frame of FMC data into memory in BRAIN format

%NOTES

clear;
close all;
clc;

%Array details - all NDT lab arrays should have a corresponding file in the
%NDT library!
array_fname = 'Imasonic 1D 64els 2.00MHz 1.50mm pitch.mat';

%DSL acquisition details
options.sample_freq = '40'; %can be 20 or 40
options.pulse_voltage = 80; %max voltage is 150
options.pulse_frequency = 2; %Frequency of excitation pulse in MHz
options.pulse_cycles = 0.5; %the number of cycles in the excitation pulse
options.pulse_active = 100; %The percentage of the excitation pulse that is active
options.time_pts = 1000;
options.sample_bits = 8; %no option to set this in DSL cfg file???
options.db_gain = 28; %max gain is ????
options.filter_no = 4; %filters not set in cfg file currently
options.prf = 2000; %maximum pulse repetition frequency in (kHz)
options.averages = 1; %no option to set this in DSL cfg file
options.gate_start = 2e-6; %gate start time in s
options.acquire_mode = 'HMC';

%Ultrasonic data (required for imaging and will be stored in exp_data file)
ph_velocity = 1450;

%Save data information if required
setupfilename ='C:\DSL FIT\Setups\DSL Test Setup.cfg';
Csetupfilename ='C:\\DSL FIT\\Setups\\DSL Test Setup.cfg';

%Display messages
echo_on = 1;    %1 - display messages, 0 - do not display messages

%-------------------------------------------------------------------------%

%Connect
if ~fn_DSL_connect(1)
    disp('Failed to connect');
    return;
end;

%Load the array file and set field in exp_data
tmp = load(array_fname);
exp_data.array = tmp.array;
no_channels = length(tmp.array.el_xc);

%Setup options and transfer to device
%Setup other parameters required by processing code
switch options.acquire_mode
    case 'SAFT'
        [options.tx_ch, options.rx_ch] = fn_set_fmc_input_matrices(no_channels, 0);
        options.rx_ch = options.tx_ch;
    case 'FMC'
        [options.tx_ch, options.rx_ch] = fn_set_fmc_input_matrices(no_channels, 0);
    case 'HMC'
        [options.tx_ch, options.rx_ch] = fn_set_fmc_input_matrices(no_channels, 1);
    case 'CSM'
        options.tx_ch = ones(1, no_channels);
        options.rx_ch = ones(1, no_channels);
end
[tx_no, rx_no] = fn_DSL_define_fmc(options.tx_ch, options.rx_ch, echo_on);
time_step = 1 / (str2double(options.sample_freq));
time_axis = [options.gate_start:time_step:options.gate_start + time_step*(options.time_pts-1)]';

%Wait for system to catch up
pause(15);

%Set detailed test options for DSL hardware
%create setup file
fn_DSL_create_cfg(setupfilename, exp_data, ph_velocity, options, echo_on);
%load onto system
fn_DSL_load_setup(Csetupfilename, echo_on);

%Collect Data
if (sqrt(length(tx_no))-floor(sqrt(length(tx_no)))) ~= 0
    is_hmc = 1; 
else
    is_hmc = 0;
end

i16time_data = fn_DSL_do_test((length(time_axis)*length(tx_no)),length(tx_no),length(time_axis),tx_no, rx_no, is_hmc, echo_on);
exp_data.time_data = double(i16time_data);
exp_data.time = time_axis;
exp_data.tx = tx_no;
exp_data.rx = rx_no;

%Disconnect
if ~fn_DSL_disconnect(1)
    disp('Failed to disconnect');
    return;
end;





