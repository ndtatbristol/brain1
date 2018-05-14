function fn_packager(root_file, package_to, options)
%SUMMARY
%   Works out file dependencies of an m-file (root_file) and copies that
%   file and all dependent ones to target directory (package_to)
%AUTHOR
%   Paul Wilcox (2008)
%USAGE
%   fn_packager(root_file, package_to, options)
%INPUTS
%   root_file - the m-file of interest
%   package_to - the directory to place files in
%   options - structured variable of options:
%       options.clear_dir[1] - empties directory first
%       options.include_ndt_files[1] - includes function files found in NDT
%       toolbox
%       options.include_matlab_files[0] - includes function files found in
%       Matlab and its toolboxes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
default_options.clear_dir = 1;
default_options.include_ndt_files = 1;
default_options.include_matlab_files = 0;
options = fn_set_default_fields(options, default_options);

%create directory if nesc
if ~exist(package_to, 'dir')
    mkdir(package_to)
end;
if options.clear_dir
    delete(fullfile(package_to, '*.*'));
end;

list = depfun(root_file, '-quiet');

matlab_toolbox_location = fullfile(matlabroot, 'toolbox');
ndt_library_location = fullfile(matlabroot, 'toolbox', 'ndt-library');

disp(' ');
disp(['Copying to ', fullfile(pwd, package_to)]);
for ii = 1:length(list)
    [location, filename, ext, dummy] = fileparts(list{ii});
    if strncmp(location, matlab_toolbox_location, length(matlab_toolbox_location))
        if strncmp(location, ndt_library_location, length(ndt_library_location)) & options.include_ndt_files
            include = 1; %include NDT files has priority
        else
            if options.include_matlab_files %not an NDT file
                include = 1;
            else
                include = 0;
            end;
        end;
    else    
        include = 1; %include anything not on toolbox path anyway
    end;
    if include
        disp(['  ', filename]);
        copyfile(list{ii}, fullfile(package_to, [filename, ext]));
    end;
end;
disp('Package complete');
disp(' ');
return;