function gpu_present = fn_test_if_gpu_present_and_working(varargin)
%SUMMARY
% Tests for presence of working GPU, including an actual calculation test
% that catches, e.g., when using remote desktop and other checks are all OK
%INPUT
% [display_results = 0] - displays result of each operation
%OUTPUT
% gpu_present - 1 if it is, 0 if not.

%--------------------------------------------------------------------------
if nargin >0
    display_results = varargin{1};
else
    display_results = 0;
end
gpu_present = 0;

%first - does the function exist (effectively checks if Matlab version is OK)
if display_results
    fprintf('Testing for existence of gpuDeviceCount function ... ');
end
if ~(exist('gpuDeviceCount') == 2)
    if display_results
        fprintf('not found\n');
    end
    return
else
    if display_results
        fprintf('OK\n');
    end
end

%second - is GPU present
if display_results
    fprintf('Testing for presence of suitable GPU device ... ');
end
if gpuDeviceCount == 0
    if display_results
        fprintf('not found\n');
    end
    return
else
    if display_results
        fprintf('OK\n');
    end
end

%third - check for error
if display_results
    fprintf('Performing test GPU operation ... ');
end
try
    a = gpuArray(rand(10));
    b = a .^ 2;
    b = gather(b);
catch
    if display_results
        fprintf('failed\n');
		fprintf(lasterr);
    end
    return
end
if display_results
    fprintf('OK\n');
end
gpu_present = 1;
end