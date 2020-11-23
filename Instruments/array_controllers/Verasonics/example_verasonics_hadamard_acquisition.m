%This example file connects to the Verasonics array controller, captures
%one frame of FMC data, optionally saves the FMC data in the exp_data
%format and generates a basic TFM image.

clc
%close all
clear all

global Trans
global Resource
global TW
global TX
global Event
global Receive
global SeqControl
global TGC
global VDASupdates
global VDAS
global TPC

RcvData=[];
TX=[];
Event=[];
Receive=[];
TPC=[];

%clear RcvData
%Array details - all NDT lab arrays should have a corresponding file in the
%NDT library!
array_fname = 'Imasonic 1D 64els 5.00MHz 0.63mm pitch.mat';
%array_fname = '1D 64els 5.00MHz 0.60mm pitch.mat';
tmp = load(array_fname);

%Micropulse acquisition details
half_matrix_capture = 0; %1 for half matrix or 0 for full matrix
test_options.pulse_voltage = 1.6; %max voltage is 96V
test_options.pulse_width = 5; %pulse width given in number of cycles
test_options.pulse_freq = tmp.array.centre_freq; %if this field is included will override array centre freq
test_options.sample_freq = 25*1e6;
test_options.time_pts = 500; %max for this setting is 3000 if doing any averaging
test_options.db_gain = 10; %max gain is 54dB
test_options.prf = 1000; %maximum pulse repetition frequency
test_options.gate_start = 5e-6;%0e-6; %gate start time in s
test_options.averages = 1;
test_options.instrument_delay = 510; %given in ns
%test_options.attenuation = 20; %given in dB/us
%test_options.attenuation_end = 3; %given in us
soft_averages = 1;

%Ultrasonic data (required for imaging and will be stored in exp_data file
ph_velocity = 6300;

%Save data information if required
save_data = 0; % 1 to save ,0 not to save
filename ='example_exp_data_file.mat';

%TFM imaging data
filter_on = 1;
centre_freq = 5e6;
half_bandwidth = 4e6;

%image size (used for all algorithms)
image_on = 1;
xrange = [-1, 1] * 40e-3;
zrange = [0, 1] * 55e-3;
pixel_size = 0.15e-3;
db_scale = 80;

%-------------------------------------------------------------------------%
%Load the array file and set field in exp_data

exp_data.array = tmp.array;

%Work out the transmit and receive sequence according to the
%number of elements in array and whether FMC or HMC
no_elements = length(exp_data.array.el_xc);
[test_options.tx_ch, test_options.rx_ch] = fn_set_fmc_input_matrices(no_elements, half_matrix_capture);

%setup hadamard matrix and its inverse
h = hadamard(2^nextpow2(no_elements+1));
s = h(2:end, 2:end);
s = s(1:no_elements, 1:no_elements);
test_options.tx_ch = s;
s_inv = inv(s);
n = size(s_inv, 2);
r = repmat([1: n ^ 2]',n, 1);
c = repmat([1: n]',n, n) + repmat([0: n - 1] * n, n ^ 2, 1);
big_inv_s = repmat(s_inv(:)',[n,1]);
big_inv_s = big_inv_s(:);
big_inv_s = sparse(r(:), c(:), big_inv_s(:), n ^ 2, n ^ 2);

[RcvProfile Control VSX_Control] = fn_set_test_options_verasonics(test_options, ph_velocity, exp_data.array);
runAcq(VSX_Control);
[result, Trans.use_volts] = setTpcProfileHighVoltage(test_options.pulse_voltage,1);
runAcq(Control(3))
pause(0.1);
Control=Control(1:2);

%Capture one set of data from Micropulse
tic;
for ii=1:soft_averages
    runAcq(VSX_Control);
    runAcq(Control);
    if ii==1
       tmp_dat=zeros(size(RcvData{1}));
   end
   tmp_dat=tmp_dat+double(RcvData{1});
end
RcvData=[];
RcvData{1}=tmp_dat./soft_averages;
[exp_data]=fn_verasonics_convert_short(Trans, Receive, RcvData);
exp_data.time_data=exp_data.time_data./test_options.averages;
exp_data.raw_data = exp_data.time_data;
exp_data.time_data = exp_data.time_data * big_inv_s;
exp_data.time=exp_data.time-test_options.instrument_delay.*1e-9-((1./(test_options.pulse_freq)).*(test_options.pulse_width/2));
time_to_capture = toc;

disp(sprintf('Data acquired in %.3f seconds', time_to_capture));

exp_data.array=tmp.array;


%Display the pulse-echo signals as waterfall plot for diagnostic purposes
kk = find(exp_data.tx == exp_data.rx);
%figure;
%plot(exp_data.time, exp_data.time_data(:, kk) + ones(size(exp_data.time)) * [1:length(kk)]);

%close the instrument
Result = hardwareClose();

%Save exp_data if required
if save_data
    save(filename, 'exp_data')
    disp('File saved');
end

%-------------------------------------------------------------------------%
%TFM imaging part
if image_on
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
    result_tfm = fn_fast_DAS2(exp_data, TFM_focal_law);
    time_to_image = toc;
    disp(sprintf('TFM image generated in %.3f seconds', time_to_image));
    
    %Display image
    figure;
    tmp = result_tfm;
    imagesc(xrange, zrange, 20 * log10(abs(tmp)/ max(max(abs(tmp))) ));%
    caxis([-db_scale, 0]);
    axis equal;
    axis tight;
    colorbar;
    title('Hadamard acquisition')
end