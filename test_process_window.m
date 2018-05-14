fn_clear;
load('test_data_set_16els_hmc.mat');
addpath('./Imaging');
h_fn_process = @fn_saft_wrapper;
h_fn_close = [];
[h_figure, h_fn_update_data, h_fn_get_options, h_fn_set_options] = gui_process_window(h_fn_process, h_fn_close, exp_data);
h_fn_update_data(exp_data);