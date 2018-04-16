function fn_prepare_brain_folders(local_folder, brain_folder, config)
try
    %try necessary in case write access blocked
    if exist(fullfile(local_folder, config.files.local_brain_path)) ~= 7
        mkdir(fullfile(local_folder, config.files.local_brain_path));
    end
    subfolders = {config.files.arrays_path, config.files.materials_path, config.files.setups_path, config.files.analysis_path, config.files.imaging_path};
    for ii = 1:length(subfolders)
        if exist(fullfile(local_folder, config.files.local_brain_path, subfolders{ii})) ~= 7
            mkdir(fullfile(local_folder, config.files.local_brain_path, subfolders{ii}));
        end
    end
catch
    warndlg('Could not create local folder structure','Warning');
end
end