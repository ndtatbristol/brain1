function available_analysis = fn_get_available_analysis(folder)
%this functions needs to have an isdeployed version that gets the actual
%function names of processes to deploy out of a file as well as a normal
%version which just bases the list on processes found in the Processes
%folder

%need to check if isdeployed is a function, as it is not available on all
%matlab installations - e.g. mu laptop version!

if isdeployed
    available_analysis(1).fn_process = @fn_2d_sizing_wrapper;
    available_analysis(2).fn_process = @fn_2d_smatrix_orig_method_wrapper;
    available_analysis(3).fn_process = @fn_calc_ang_dep_vel_wrapper;
    available_analysis(4).fn_process = @fn_estimate_attenuation_wrapper;
else
    if ~iscell(folder)
        folder = {folder};
    end
    %add all subfolders of specified directory to path
    n = 1; 
    for fi =1:length(folder)
        tmp = dir(fullfile(folder {fi}, 'fn_*_wrapper.m'));
        if ~isempty(tmp)
            addpath(genpath(folder{fi}));
        end
        for ii = 1:length(tmp)
            available_analysis(n).fn_process = str2func(tmp(ii).name(1:end-2));
%             info = available_analysis(n).fn_process([], []);
%             available_analysis(n).name = info.name;
            n = n + 1;
        end
    end
end
fprintf('Available analysis functions\n');
for ii = 1:length(available_analysis)
    info = available_analysis(ii).fn_process([], []);
    available_analysis(ii).name = info.name;
    fprintf(['  %2i. ', available_analysis(ii).name, '\n'], ii);
end
end