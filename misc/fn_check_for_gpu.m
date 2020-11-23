function result = fn_check_for_gpu
%SUMMARY
%   Quick checks for presence of GPU. For more comprehensive test with some
%   diagnostics, use fn_test_if_gpu_present_and_working instead.
result = (exist('gpuDeviceCount', 'file') == 2) && (gpuDeviceCount > 0);
end