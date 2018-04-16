fn_clear;
addpath('..');
load('../Instruments/Emulator data/64_els_exp_data_for_emulator.mat');
% density = 2700;
% exp_data.centre_freq = exp_data.input_freq;
% exp_data.bandwidth = exp_data.input_freq / 4; %er?
% exp_data.array.el_width = abs(exp_data.array.el_x2(1) - exp_data.array.el_xc(1)) * 2;
% exp_data.array.el_pitch = abs(exp_data.array.el_xc(2) - exp_data.array.el_xc(1));
% exp_data.matl_props.xi = 2;%ratio L to S speed
% exp_data.matl_props.mu = density * (exp_data.ph_velocity/exp_data.matl_props.xi)^2;
% % 
% exp_data.time_data = exp_data.time_data(1:500,:);
% exp_data.time = exp_data.time(1:500);
% 
% region.x = [-10, 0] * 1e-3;
% region.z = [15, 25] * 1e-3;
% tic;
% s = fn_1Darray_Smatrix_extraction_v1(exp_data,region);
% toc
% imagesc(s.phi * 180 / pi, s.phi * 180 / pi,abs(s.m));

options.select = [[-20; 20], [0; 60]] * 1e-3;
pts = 100;
data.x = linspace(options.select(1,1), options.select(2,1), pts);
data.z = linspace(options.select(1,2), options.select(2,2), pts);
[mesh.x, mesh.z] = meshgrid(data.x, data.z);

TFM_focal_law = fn_calc_tfm_focal_law2(exp_data, mesh);
TFM_focal_law.interpolation_method = 'linear';
TFM_focal_law.filter_on = 1;
TFM_focal_law.filter = fn_calc_filter(exp_data.time, 5e6, 5e6);

data.f = fn_fast_DAS2(exp_data, TFM_focal_law);

figure;
tmp = abs(data.f);
tmp = 20 * log10(tmp / max(max(tmp)));
imagesc(data.x * 1e3, data.z * 1e3, tmp);axis ij; axis tight;
caxis([-40, 0]);

options.select = [[-10; 0], [15; 25]] * 1e-3;
fn_2d_smatrix_orig_method_xxx(exp_data, data, options);



