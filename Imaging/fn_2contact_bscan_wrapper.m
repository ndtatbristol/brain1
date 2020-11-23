function varargout = fn_2contact_bscan_wrapper(exp_data, options, mode)
%SUMMARY
%   This performs a simple bscan

%USAGE (depending on value of mode argument)
%   initial_info = fn_basic_wrapper([], [], 'return_name_only')
%   extended_info = fn_basic_wrapper(exp_data, [], 'return_info_only')
%   [data, options_with_precalcs] = fn_basic_wrapper(exp_data, options, 'recalc_and_process')
%   data = fn_tfm_wrapper(exp_data, options_with_precalcs, 'process_only')

% default_options.options_changed = 1; %this enables recurring data (e.g. distances to field points to be calculated as a field in options and only recalculated if options_changed = 1)

%the following is the data to allow the processing parameters to be
%displayed and edited in the GUI as well as the default values.
name = 'Contact B-scan'; %name of process that appears on menu
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

global GPU_PRESENT

if isfield(exp_data, 'material')
    options_with_precalcs.original_material = exp_data.material;
elseif isfield(exp_data, 'vel_elipse')
    lebedev_quality = 3;
    options_with_precalcs.original_material.vel_spherical_harmonic_coeffs = fn_spherical_harmonics_for_elliptical_profile(exp_data.vel_elipse(1), exp_data.vel_elipse(1), exp_data.vel_elipse(2), lebedev_quality);
elseif isfield(exp_data, 'ph_velocity')
    options_with_precalcs.original_material.vel_spherical_harmonic_coeffs = exp_data.ph_velocity; %legacy files
else
    error('No valid velocity description found');
end

%check style of data and array
using_2d_array = any(exp_data.array.el_yc);
data_is_csm = length(unique(exp_data.tx)) == 1;


%set up grid and image axes
if using_2d_array
    options_with_precalcs.data.x = [-options.x_size / 2: options.pixel_size: options.x_size / 2] + options.x_offset;
    options_with_precalcs.data.y = [-options.y_size / 2: options.pixel_size: options.y_size / 2] + options.y_offset;
    options_with_precalcs.data.z = [0: options.pixel_size: options.z_size] + options.z_offset;
    [tmp_mesh.x, tmp_mesh.y, tmp_mesh.z] = meshgrid(options_with_precalcs.data.x, options_with_precalcs.data.y, options_with_precalcs.data.z);
else
    options_with_precalcs.data.x = [-options.x_size / 2: options.pixel_size: options.x_size / 2] + options.x_offset;
    options_with_precalcs.data.z = [0: options.pixel_size: options.z_size] + options.z_offset;
    [tmp_mesh.x, tmp_mesh.z] = meshgrid(options_with_precalcs.data.x, options_with_precalcs.data.z);
end

calc_focal_law_options.aperture_size_metres = options.aperture_size_metres;

%focal law calculation
if data_is_csm
    calc_focal_law_options.transmit_mode = 'CSM';
end

options_with_precalcs.focal_law = fn_calc_bscan_focal_law3(exp_data, tmp_mesh, calc_focal_law_options);

options_with_precalcs.focal_law.interpolation_method = lower(options.interpolation_method);
options_with_precalcs.focal_law.filter_on = options.filter_on;
options_with_precalcs.focal_law.filter = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_half_bandwidth / 2);

%geometry lines for plot
dx = options_with_precalcs.aperture_size_metres / 2;
options_with_precalcs.geom.lines(1).x = [min(exp_data.array.el_xc) + dx, max(exp_data.array.el_xc) - dx; ...
    min(exp_data.array.el_xc) + dx, max(exp_data.array.el_xc) - dx];
options_with_precalcs.geom.lines(1).y = zeros(2,2);
options_with_precalcs.geom.lines(1).z = [0, 0; ones(1,2) * max(options_with_precalcs.data.z)];
options_with_precalcs.geom.lines(1).style = ':';
options_with_precalcs.geom.array = fn_get_array_geom_for_plots(exp_data.array);

%GPU check
% if (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0) && (options_with_precalcs.use_gpu_if_available)
%     if ~isfield(options_with_precalcs.focal_law, 'thread_size')
%         gpu_han=gpuDevice;
%         options_with_precalcs.focal_law.thread_size=gpu_han.MaxThreadsPerBlock;
%     end
% end
end

%--------------------------------------------------------------------------

function data = fn_process_using_precalcs(exp_data, options_with_precalcs)
%put the actual imaging calculations here, making use of pre-calculated
%values in the options_with_precalcs fields if required.

if isfield(options_with_precalcs, 'use_gpu')
    use_gpu = options_with_precalcs.use_gpu;
else
    use_gpu = 0;
end

%copy output output coordinates
data.x = options_with_precalcs.data.x;
if isfield(options_with_precalcs.data, 'y')
    data.y = options_with_precalcs.data.y;
end;
data.z = options_with_precalcs.data.z;

%the actual calculation
data.f = fn_fast_DAS3(exp_data, options_with_precalcs.focal_law, use_gpu);
if  isfield(options_with_precalcs.focal_law,'thread_size')
   data.f=(gather(data.f));
end
data.geom = options_with_precalcs.geom;
end

%--------------------------------------------------------------------------

function info = fn_return_info(exp_data)
global GPU_PRESENT

using_2d_array = ~isempty(exp_data) && any(exp_data.array.el_yc); %PW - added this as it is tested for at numerous points below

if isfield(exp_data.array, 'centre_freq') %PW -used in various places below to set initial values
    default_freq = exp_data.array.centre_freq;
else
    default_freq = 5e6;
end
default_half_bandwidth = 2;

if isfield(exp_data, 'vel_elipse') %for legacy files, the spherical harmonic coeffs are not defined for ellipse at this point, so need to read default values from legacy info
    v = exp_data.ph_velocity;
    v_x = exp_data.vel_elipse(1);
    v_y = exp_data.vel_elipse(1);
    v_z = exp_data.vel_elipse(2);
elseif (isfield(exp_data, 'material') && isfield(exp_data.material, 'vel_spherical_harmonic_coeffs'))
    [v, v_x, v_y, v_z] = fn_get_nominal_velocity(exp_data.material.vel_spherical_harmonic_coeffs);
elseif isfield(exp_data, 'ph_velocity')
    v = exp_data.ph_velocity;
    v_x = v;
    v_y = v;
    v_z = v;
else
    error('No valid velocity description found');
end

info.display_options.axis_equal = 1;
info.display_options.x_axis_sf = 1e3;
info.display_options.z_axis_sf = 1e3;

im_sz_z = max(exp_data.time) * v / 2;
im_sz_xy = max([...
    max(exp_data.array.el_xc) - min(exp_data.array.el_xc), ...
    max(exp_data.array.el_yc) - min(exp_data.array.el_yc)]);

if using_2d_array
    info.fn_display = @gui_3d_plot_panel;
    info.display_options.interpolation = 0;
    info.display_options.y_axis_sf = 1e3;
    no_pixels = 30;
    default_pixel_size = max([im_sz_xy, im_sz_z]) / no_pixels;
    round_to = 0.25e-3;
    array_size = max(exp_data.array.el_xc(:)) - min(exp_data.array.el_xc(:));
else
    info.fn_display = @gui_2d_plot_panel;
    info.display_options.interpolate = 0;
    if GPU_PRESENT 
        max_freq = default_freq * (1 + default_half_bandwidth / 2);
        default_pixel_size = (v ./ max_freq) / 4; %divide by four is to account for nyquist frequency and out and back path length;
        round_to = 0.01e-3;
    else
        no_pixels = 100;
        default_pixel_size = max([im_sz_xy, im_sz_z]) / no_pixels;
        round_to = 0.25e-3;
    end
    array_size = mean([max(exp_data.array.el_xc(:)) - min(exp_data.array.el_xc(:)), max(exp_data.array.el_yc(:)) - min(exp_data.array.el_yc(:))]);
end
round(default_pixel_size / round_to) * round_to;

default_aperture_size = array_size / 8;

if GPU_PRESENT
    info.display_options.gpu = 1;
end

if isempty(exp_data)
    varargout{1} = [];
    varargout{2} = info;
    return %this is the exit point if exp_data does not exist
end

%--------------------------------------------------------------------------
%Image size and resolution
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

if using_2d_array
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
info.options_info.pixel_size.default = default_pixel_size;
info.options_info.pixel_size.type = 'double';
info.options_info.pixel_size.constraint = [1e-6, 1];
info.options_info.pixel_size.multiplier = 1e-3;

info.options_info.filter_on.label = 'Filter';
info.options_info.filter_on.type = 'bool';
info.options_info.filter_on.constraint = {'On', 'Off'};
info.options_info.filter_on.default = 1;

info.options_info.centre_freq.label = 'Filter freq (MHz)';
info.options_info.centre_freq.default = default_freq;
info.options_info.centre_freq.type = 'double';
info.options_info.centre_freq.constraint = [0.1, 20e6];
info.options_info.centre_freq.multiplier = 1e6;

info.options_info.frac_half_bandwidth.label = 'Percent b/width';
info.options_info.frac_half_bandwidth.default = default_half_bandwidth;
info.options_info.frac_half_bandwidth.type = 'double';
info.options_info.frac_half_bandwidth.constraint = [0.01, 10];
info.options_info.frac_half_bandwidth.multiplier = 0.01;

info.options_info.const_velocity.label = 'Velocity (m/s)';
info.options_info.const_velocity.default = v;
info.options_info.const_velocity.type = 'double';
info.options_info.const_velocity.constraint = [1, 20000];
info.options_info.const_velocity.multiplier = 1;

info.options_info.aperture_size_metres.label = 'Aperture size (mm)';
info.options_info.aperture_size_metres.default = default_aperture_size;
info.options_info.aperture_size_metres.type = 'double';
info.options_info.aperture_size_metres.constraint = [0.0001, 10];
info.options_info.aperture_size_metres.multiplier = 1e-3;

info.options_info.interpolation_method.label = 'Interpolation';
info.options_info.interpolation_method.default = 'Nearest';
info.options_info.interpolation_method.type = 'constrained';
info.options_info.interpolation_method.constraint = {'Linear', 'Nearest'};

end
