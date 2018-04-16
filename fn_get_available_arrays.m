function available_arrays = fn_get_available_arrays(folder)
if ~iscell(folder)
    folder = {folder};
end
jj = 1;
available_arrays(jj).name = 'No array selected';
available_arrays(jj).array = [];
jj = 2;
fprintf('Available arrays\n');
for fi =1:length(folder)
    tmp = dir(fullfile(folder{fi}, '*.mat'));
    for ii = 1:length(tmp)
        load(fullfile(folder{fi}, tmp(ii).name));
        if exist('array', 'var')
            available_arrays(jj).array = array;
            [dummy, name, ext] = fileparts(tmp(ii).name);
            available_arrays(jj).name = name;
            fprintf(['  %2i. ', available_arrays(jj).name, '\n'], jj - 1);
            jj = jj + 1;
        end
    end
end
end