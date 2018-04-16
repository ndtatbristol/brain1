function available_processes = fn_get_available_processes(pt)
%this functions needs to have an isdeployed version that gets the actual
%function names of processes to deploy out of a file as well as a normal
%version which just bases the list on processes found in the Processes 
%folder

%need to check if isdeployed is a function, as it is not available on all
%matlab installations - e.g. mu laptop version!

% if isdeployed
%     available_processes(1).fn_process = @fn_saft_wrapper;
%     available_processes(2).fn_process = @fn_tfm_wrapper;
%     available_processes(3).fn_process = @fn_bscan_wrapper;
% else
    %add all subfolders of specified directory to path
    tmp = dir([pt,filesep, 'fn_*_wrapper.m']);
    if isempty(tmp)
        available_processes = [];
        return;
    end
    addpath(genpath(pt));
    for ii = 1:length(tmp)
        available_processes(ii).fn_process = str2func(tmp(ii).name(1:end-2));
        [dummy, dummy, info] = available_processes(ii).fn_process([], []);
        available_processes(ii).name = info.name;
    end
% end
fprintf('Available imaging\n');
for ii = 1:length(available_processes)
    fprintf(['  %2i. ', available_processes(ii).name, '\n'], jj - 1);
end
end