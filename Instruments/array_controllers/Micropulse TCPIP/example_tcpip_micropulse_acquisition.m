%This example file connects to the Micropulse array controller, captures
%one frame of FMC data, optionally saves the FMC data in the exp_data
%format and generates a basic TFM image.

clc;
close all;

if exist('tcpip_obj') && isa(tcpip_obj, 'tcpip')
    fclose(tcpip_obj);
    delete(tcpip_obj);
    clear('tcpip_obj');
    disp('Clearing');
end

%clear all;

%make tcp/ip object global. Dirty fix, will be improved with object
%orientated version.
global tcpip_obj


%Array details - all NDT lab arrays should have a corresponding file in the
%NDT library!
array_fname = 'Imasonic 1D 64els 5.00MHz 0.60mm pitch.mat';

%Micropulse connection details
ip_address = '10.1.1.2';
port_no = 1067;
echo_on =0;

%Micropulse reset details
full_reset = 0;

%Micropulse acquisition details
half_matrix_capture = 1; %1 for half matrix or 0 for full matrix
test_options.sample_freq = 25e6; %can be 25MHz, 50MHz or 100MHz
test_options.pulse_voltage = 200; %max voltage is 200V
test_options.pulse_width = 80e-9;
test_options.time_pts = 3000; %max for this setting is 3000 if doing any averaging
test_options.sample_bits = 8; %8, 10, 12, 16
test_options.db_gain = 20; %max gain is 70dB
test_options.filter_no = 4; %1:5-10MHz 2:2-10MHz 3:0.75-5MHz 4:0.75-20MHz
test_options.prf = 2000; %maximum pulse repetition frequency
test_options.averages = 0;
test_options.gate_start = 0; %gate start time in s

%Ultrasonic data (required for imaging and will be stored in exp_data file
ph_velocity = 6000;

%Save data information if required
save_data = 0; % 1 to save ,0 not to save
filename ='example_exp_data_file.mat';

%TFM imaging data
filter_on = 1;
centre_freq = 5e6;
half_bandwidth = 4e6;

%image size (used for all algorithms)
xrange = [-1, 1] * 40e-3;
zrange = [0, 1] * 40e-3;
pixel_size = 0.5e-3;
db_scale = 40;

%-------------------------------------------------------------------------%

%Connect
if ~fn_ag_connect_tcpip(ip_address, port_no, 1)
    disp('Failed to connect');
    return;
end;

%Reset if required
if full_reset
    reset_result = fn_ag_reset_tcpip(echo_on);
end;

%Load the array file and set field in exp_data
tmp = load(array_fname);
exp_data.array = tmp.array;

%Work out the transmit and receive sequence according to the
%number of elements in array and whether FMC or HMC
no_elements = length(exp_data.array.el_xc);
[test_options.tx_ch, test_options.rx_ch] = fn_set_fmc_input_matrices(no_elements, half_matrix_capture);

%Send detailed test_options to Micropulse
[exp_data.tx, exp_data.rx] = fn_ag_set_test_options_tcpip(test_options, echo_on);

pause(0.2)
%Capture one set of data from Micropulse
tic;
exp_data.time_data = fn_ag_do_test_tcpip(echo_on);
time_to_capture = toc;
disp(sprintf('Data acquired in %.3f seconds', time_to_capture));

%Define the time base in exp_data
time_step = 1 / test_options.sample_freq;
exp_data.time = [test_options.gate_start:time_step:test_options.gate_start + time_step*(test_options.time_pts-1)]';

%Add other fields to exp_data
exp_data.test_options = test_options;
exp_data.ph_velocity = ph_velocity;

%Display the pulse-echo signals as waterfall plot for diagnostic purposes
kk = find(exp_data.tx == exp_data.rx);
figure;
plot(exp_data.time, exp_data.time_data(:, kk) + ones(size(exp_data.time)) * [1:length(kk)]);
%close the instrument 
fclose(tcpip_obj)
delete(tcpip_obj)
clear tcpip_obj

%Save exp_data if required
if save_data
    save(filename, 'exp_data')
    disp('File saved');
end

%-------------------------------------------------------------------------%
%TFM imaging part

%Set up image coordinates
x = linspace(xrange(1), xrange(2), ceil((xrange(2) - xrange(1)) / pixel_size));
z = linspace(zrange(1), zrange(2), ceil((zrange(2) - zrange(1)) / pixel_size));
[mesh.x, mesh.z] = meshgrid(x, z);

%Set up filter
filter = fn_calc_filter(exp_data.time, centre_freq, half_bandwidth);

TFM_focal_law = fn_calc_tfm_focal_law2(exp_data, mesh);
TFM_focal_law.filter_on = filter_on;
TFM_focal_law.filter = fn_calc_filter(exp_data.time, centre_freq, half_bandwidth);

%Actual imaging calculation
tic;
result_tfm = gather(fn_fast_DAS3(exp_data, TFM_focal_law));
time_to_image = toc;
disp(sprintf('TFM image generated in %.3f seconds', time_to_image));

%Display image
figure;
tmp = result_tfm;
imagesc(xrange, zrange, 20 * log10(abs(tmp) / max(max(abs(tmp)))));
caxis([-db_scale, 0]);
axis equal;
axis tight;
colorbar;