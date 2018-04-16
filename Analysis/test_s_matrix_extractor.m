fn_clear;
addpath('../arrays/2d-reversible-imaging');
load('./Instruments/64_els_exp_data_for_emulator.mat');
% density = 2700;
% exp_data.centre_freq = exp_data.input_freq;
% exp_data.bandwidth = exp_data.input_freq / 4; %er?
% exp_data.array.el_width = abs(exp_data.array.el_x2(1) - exp_data.array.el_xc(1)) * 2;
% exp_data.array.el_pitch = abs(exp_data.array.el_xc(2) - exp_data.array.el_xc(1));
% exp_data.matl_props.xi = 2;%ratio L to S speed
% exp_data.matl_props.mu = density * (exp_data.ph_velocity/exp_data.matl_props.xi)^2;
% % 
exp_data.time_data = exp_data.time_data(1:500,:);
exp_data.time = exp_data.time(1:500);
% 
% region.x = [-10, 0] * 1e-3;
% region.z = [15, 25] * 1e-3;
% tic;
% s = fn_1Darray_Smatrix_extraction_v1(exp_data,region);
% toc
% imagesc(s.phi * 180 / pi, s.phi * 180 / pi,abs(s.m));

data = [];
options.select = [[-10; 0], [-15; 25]] * 1e-3;
fn_2d_smatrix_wrapper(exp_data, data, options);