function result = fn_check_for_gpu
error('Do not use this function! It does not do a complete enough check for a working GPU. Replace with fn_test_if_gpu_present_and_working');

%SUMMARY
%   Quick checks for presence of GPU. For more comprehensive test with some
%   diagnostics, use fn_test_if_gpu_present_and_working instead.
result = (exist('gpuDeviceCount', 'file') == 2) && (gpuDeviceCount > 0);
end