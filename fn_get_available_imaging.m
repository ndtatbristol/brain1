function available_imaging = fn_get_available_imaging(folder)

if isdeployed
    disp('fn_get_available_imaging - deployed version');
    available_imaging(1).fn_process = @fn_1contact_tfm_wrapper;
    available_imaging(2).fn_process = @fn_2immersion_tfm3_wrapper;
    available_imaging(3).fn_process = @fn_3contact_bscan_wrapper;
    available_imaging(4).fn_process = @fn_4contact_sector_scan_wrapper;
    available_imaging(5).fn_process = @fn_5contact_comp_wrapper;
    available_imaging(6).fn_process = @fn_6adaptive_oblique_tfm_wrapper;
    available_imaging(7).fn_process = @fn_10ascan_wrapper;
else
    if ~iscell(folder)
        folder = {folder};
    end
    %add all subfolders of specified directory to path
    n = 1;
    for fi = 1:length(folder)
        tmp = dir(fullfile(folder{fi}, 'fn_*_wrapper.m'));
        if ~isempty(tmp)
            addpath(genpath(folder{fi}));
        end
        for ii = 1:length(tmp)
            available_imaging(n).fn_process = str2func(tmp(ii).name(1:end-2));
            n = n + 1;
        end
    end
end
fprintf('Available imaging\n');

for ii = 1:length(available_imaging)
    available_imaging(ii).name = available_imaging(ii).fn_process([], [], 'return_name_only');
    fprintf(['  %2i. ', available_imaging(ii).name, '\n'], ii);
end
end