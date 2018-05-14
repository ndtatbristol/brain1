function available_materials = fn_get_available_materials(folder)
available_materials = [];
if ~iscell(folder)
    folder = {folder};
end;
jj = 1;
available_materials(jj).material = [];
available_materials(jj).name = 'No material selected';
jj = 2;
fprintf('Available materials\n');
for fi = 1:length(folder)
    tmp = dir(fullfile(folder{fi}, '*.mat'));
    for ii = 1:length(tmp)
        tmp2 = load(fullfile(folder{fi}, tmp(ii).name));
        if isfield(tmp2, 'material')
            available_materials(jj).material = tmp2.material;
            [dummy, name, ext] = fileparts(tmp(ii).name);
            available_materials(jj).name = name;
            fprintf(['  %2i. ', available_materials(jj).name, '\n'], jj - 1);
            jj = jj + 1;
        end
    end
end
end