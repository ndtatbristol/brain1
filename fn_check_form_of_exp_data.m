function [using_2d_array, data_is_csm] = fn_check_form_of_exp_data(exp_data)
%SUMMARY
%   Utility functiont that returns flags if exp_data is from 2D array or
%   acquired using CSM
%INPUTS
%   exp_data - exp_data structure
%OUTPUTS
%   using_2d_array - 1 if data is from 2d array, 0 otherwise
%   data_is_csm - 1 if data is from CSM acquisition, 0 otherwise
%--------------------------------------------------------------------------
using_2d_array = any(exp_data.array.el_yc);
data_is_csm = length(unique(exp_data.tx)) == 1;
end