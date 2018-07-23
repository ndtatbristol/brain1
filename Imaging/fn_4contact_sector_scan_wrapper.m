function varargout = fn_contact_sector_scan_wrapper(exp_data, options, mode)
%SUMMARY
%   This performs a simple sector scan

%USAGE (depending on value of mode argument)
%   initial_info = fn_basic_wrapper([], [], 'return_name_only')
%   extended_info = fn_basic_wrapper(exp_data, [], 'return_info_only')
%   [data, options_with_precalcs] = fn_basic_wrapper(exp_data, options, 'recalc_and_process')
%   data = fn_tfm_wrapper(exp_data, options_with_precalcs, 'process_only')

% default_options.options_changed = 1; %this enables recurring data (e.g. distances to field points to be calculated as a field in options and only recalculated if options_changed = 1)

%the following is the data to allow the processing parameters to be
%displayed and edited in the GUI as well as the default values.
name = 'Contact sector-scan'; %name of process that appears on menu
switch mode
    case 'return_name_only'
        varargout{1} = name;
        return;
        
    case 'return_info_only'
        info = fn_return_info(exp_data);
        info.name = name;
        varargout{1} = info;
        return;
        
    case 'recalc_and_process'
        options_with_precalcs = fn_return_options_with_precalcs(exp_data, options);
        data = fn_process_using_precalcs(exp_data, options_with_precalcs);
        varargout{1} = data;
        varargout{2} = options_with_precalcs;

    case 'process_only'
        data = fn_process_using_precalcs(exp_data, options);
        varargout{1} = data;

end
end

%--------------------------------------------------------------------------

function options_with_precalcs = fn_return_options_with_precalcs(exp_data, options)
options_with_precalcs = options; %need this line as initial options 

exp_data.ph_velocity = options_with_precalcs.ph_velocity;

%set up grid and image axes
if any(exp_data.array.el_yc)
    options_with_precalcs.data.x = [-options.x_size / 2: options.pixel_size: options.x_size / 2] + options.x_offset;
    options_with_precalcs.data.y = [-options.y_size / 2: options.pixel_size: options.y_size / 2] + options.y_offset;
    options_with_precalcs.data.z = [0: options.pixel_size: options.z_size] + options.z_offset;
    [tmp_mesh.x, tmp_mesh.y, tmp_mesh.z] = meshgrid(options_with_precalcs.data.x, options_with_precalcs.data.y, options_with_precalcs.data.z);
else
    options_with_precalcs.data.x = [-options.x_size / 2: options.pixel_size: options.x_size / 2] + options.x_offset;
    options_with_precalcs.data.z = [0: options.pixel_size: options.z_size] + options.z_offset;
    [tmp_mesh.x, tmp_mesh.z] = meshgrid(options_with_precalcs.data.x, options_with_precalcs.data.z);
end

%calc focal law -  different if data is CSM!
data_is_csm = length(unique(exp_data.tx)) == 1;
calc_focal_law_options.aperture_size_metres = options.aperture_size_metres;
calc_focal_law_options.angle_limit = options.angle_limit;
if data_is_csm
    warndlg('Cannot generate sector scan with CSM data', 'Warning');
    options_with_precalcs.focal_law = [];
    return;
else
    options_with_precalcs.focal_law = fn_calc_sector_scan_focal_law2(exp_data, tmp_mesh, calc_focal_law_options);
end
options_with_precalcs.focal_law.interpolation_method = lower(options.interpolation_method);
options_with_precalcs.focal_law.filter_on = options.filter_on;
options_with_precalcs.focal_law.filter = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_half_bandwidth / 2);

%geometry lines for plot
dx = options_with_precalcs.aperture_size_metres / 2;
% options_with_precalcs.geom.lines(1).x = [min(exp_data.array.el_xc) + dx, max(exp_data.array.el_xc) - dx; ...
%     min(exp_data.array.el_xc) + dx, max(exp_data.array.el_xc) - dx];
% options_with_precalcs.geom.lines(1).y = zeros(2,2);
% options_with_precalcs.geom.lines(1).z = [0, 0; ones(1,2) * max(options_with_precalcs.data.z)];
% options_with_precalcs.geom.lines(1).style = ':';
options_with_precalcs.geom.array = fn_get_array_geom_for_plots(exp_data.array);

%GPU check - need to add disable key as well
if (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0) && (options_with_precalcs.use_gpu_if_available)
    if ~isfield(options_with_precalcs.focal_law, 'thread_size')
        gpu_han=gpuDevice;
        options_with_precalcs.focal_law.thread_size=gpu_han.MaxThreadsPerBlock;
    end
end
end

%--------------------------------------------------------------------------

function data = fn_process_using_precalcs(exp_data, options_with_precalcs)
%put the actual imaging calculations here, making use of pre-calculated
%values in the options_with_precalcs fields if required.

if isempty(options_with_precalcs.focal_law)
    data = [];
    return;
end

%copy output output coordinates
data.x = options_with_precalcs.data.x;
if isfield(options_with_precalcs.data, 'y')
    data.y = options_with_precalcs.data.y;
end;
data.z = options_with_precalcs.data.z;

%the actual calculation
data.f = fn_fast_DAS3(exp_data, options_with_precalcs.focal_law, options_with_precalcs.use_gpu_if_available);
if  isfield(options_with_precalcs.focal_law,'thread_size')
   data.f=(gather(data.f));
end
data.geom = options_with_precalcs.geom;
end

%--------------------------------------------------------------------------

function info = fn_return_info(exp_data)
if ~isempty(exp_data) && any(exp_data.array.el_yc)
    info.fn_display = @gui_3d_plot_panel;
    info.display_options.interpolation = 0;
    no_pixels = 30;
else
    info.fn_display = @gui_2d_plot_panel;
    info.display_options.interpolation = 0;
    no_pixels = 100;
end
info.display_options.axis_equal = 1;
info.display_options.x_axis_sf = 1e3;
info.display_options.y_axis_sf = 1e3;
info.display_options.z_axis_sf = 1e3;

im_sz_z = max(exp_data.time) * exp_data.ph_velocity / 2;
im_sz_xy = im_sz_z * sqrt(2);
info.options_info.x_size.label = 'X size (mm)';
info.options_info.x_size.default = im_sz_xy;
info.options_info.x_size.type = 'double';
info.options_info.x_size.constraint = [1e-3, 10];
info.options_info.x_size.multiplier = 1e-3;

info.options_info.x_offset.label = 'X offset (mm)';
info.options_info.x_offset.default = 0;
info.options_info.x_offset.type = 'double';
info.options_info.x_offset.constraint = [-10, 10];
info.options_info.x_offset.multiplier = 1e-3;

if any(exp_data.array.el_yc)
    info.options_info.y_size.label = 'Y size (mm)';
    info.options_info.y_size.default = im_sz_xy;
    info.options_info.y_size.type = 'double';
    info.options_info.y_size.constraint = [1e-3, 10];
    info.options_info.y_size.multiplier = 1e-3;
    
    info.options_info.y_offset.label = 'Y offset (mm)';
    info.options_info.y_offset.default = 0;
    info.options_info.y_offset.type = 'double';
    info.options_info.y_offset.constraint = [-10, 10];
    info.options_info.y_offset.multiplier = 1e-3;
end

info.options_info.z_size.label = 'Z size (mm)';
info.options_info.z_size.default = im_sz_z;
info.options_info.z_size.type = 'double';
info.options_info.z_size.constraint = [1e-3, 10];
info.options_info.z_size.multiplier = 1e-3;

info.options_info.z_offset.label = 'Z offset (mm)';
info.options_info.z_offset.default = 0;
info.options_info.z_offset.type = 'double';
info.options_info.z_offset.constraint = [-10, 10];
info.options_info.z_offset.multiplier = 1e-3;

info.options_info.pixel_size.label = 'Pixel size (mm)';
info.options_info.pixel_size.default = round(max([im_sz_xy, im_sz_z]) / no_pixels * 1e3*4)/(1e3*4);
info.options_info.pixel_size.type = 'double';
info.options_info.pixel_size.constraint = [1e-6, 1];
info.options_info.pixel_size.multiplier = 1e-3;

info.options_info.filter_on.label = 'Filter';
info.options_info.filter_on.type = 'bool';
info.options_info.filter_on.constraint = {'On', 'Off'};
info.options_info.filter_on.default = 1;

info.options_info.centre_freq.label = 'Filter freq (MHz)';
if isfield(exp_data.array, 'centre_freq')
    info.options_info.centre_freq.default = exp_data.array.centre_freq;
else
    info.options_info.centre_freq.default = 5e6;
end;
info.options_info.centre_freq.type = 'double';
info.options_info.centre_freq.constraint = [0.1, 20e6];
info.options_info.centre_freq.multiplier = 1e6;

info.options_info.frac_half_bandwidth.label = 'Percent b/width';
info.options_info.frac_half_bandwidth.default = 2;
info.options_info.frac_half_bandwidth.type = 'double';
info.options_info.frac_half_bandwidth.constraint = [0.01, 10];
info.options_info.frac_half_bandwidth.multiplier = 0.01;

info.options_info.ph_velocity.label = 'Velocity (m/s)';
info.options_info.ph_velocity.default = exp_data.ph_velocity;
info.options_info.ph_velocity.type = 'double';
info.options_info.ph_velocity.constraint = [1, 20000];
info.options_info.ph_velocity.multiplier = 1;

info.options_info.aperture_size_metres.label = 'Aperture size (mm)';
info.options_info.aperture_size_metres.default = 0;
info.options_info.aperture_size_metres.type = 'double';
info.options_info.aperture_size_metres.constraint = [0, 10];
info.options_info.aperture_size_metres.multiplier = 1e-3;

info.options_info.angle_limit.label = 'Angle limit (degrees)';
info.options_info.angle_limit.default = 0;
info.options_info.angle_limit.type = 'double';
info.options_info.angle_limit.constraint = [0, pi / 2];
info.options_info.angle_limit.multiplier = pi / 180;

info.options_info.interpolation_method.label = 'Interpolation';
info.options_info.interpolation_method.default = 'Nearest';
info.options_info.interpolation_method.type = 'constrained';
info.options_info.interpolation_method.constraint = {'Linear', 'Nearest'};

% info.options_info.use_gpu_if_available.label = 'Use GPU if available';
% info.options_info.use_gpu_if_available.type = 'bool';
% info.options_info.use_gpu_if_available.constraint = {'On', 'Off'};
% info.options_info.use_gpu_if_available.default = 1;
end
