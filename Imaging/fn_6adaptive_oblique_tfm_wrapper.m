function varargout = fn_6adaptive_oblique_tfm_wrapper(exp_data, options, mode)
%SUMMARY
%   Immersion TFM for 1D array obliquely positioned over flat surface the
%   location of which is found from data or specified. Main difference to
%   general immersion code is that (a) only flat surfaces are handled and
%   (b) array position is offset and rotated relative to coordinate origin

%USAGE (depending on value of mode argument)
%   initial_info = fn_basic_wrapper([], [], 'return_name_only')
%   extended_info = fn_basic_wrapper(exp_data, [], 'return_info_only')
%   [data, options_with_precalcs] = fn_basic_wrapper(exp_data, options, 'recalc_and_process')
%   data = fn_tfm_wrapper(exp_data, options_with_precalcs, 'process_only')

% default_options.options_changed = 1; %this enables recurring data (e.g. distances to field points to be calculated as a field in options and only recalculated if options_changed = 1)

%the following is the data to allow the processing parameters to be
%displayed and edited in the GUI as well as the default values.
name = 'Oblique TFM (beta)'; %name of process that appears on menu
%force recalc of focal law if in surface measuring mode
if strcmp(mode, 'process_only') && isfield(options, 'surface_type') && strcmp(options.surface_type, '|M|easured')% && isfield(options, 'show_couplant_only') && ~options.show_couplant_only
    mode = 'recalc_and_process';
end
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
        if isempty(options_with_precalcs)
            data = []; %catch for 2D array which are not supported
        else
            data = fn_process_using_precalcs(exp_data, options_with_precalcs);
        end
        varargout{1} = data;
        varargout{2} = options_with_precalcs;
        
    case 'process_only'
        data = fn_process_using_precalcs(exp_data, options);
        varargout{1} = data;
        
end
end

%--------------------------------------------------------------------------

function options_with_precalcs = fn_return_options_with_precalcs(exp_data, options)
options_with_precalcs = options; %need this line as initial options are copied into options_with_precalcs

exp_data.ph_velocity = options_with_precalcs.ph_velocity;


%set up grid and image axes
if fn_return_dimension_of_array(exp_data.array) == 2
    options_with_precalcs = [];
    warndlg('2D arrays not yet supported','Warning')
    return;%
end

data_is_csm = length(unique(exp_data.tx)) == 1;
if data_is_csm
    options_with_precalcs = [];
    warndlg('CSM data not yet supported','Warning');
    return
end

options.x_size = max([options.x_size, options.pixel_size * 4]);
options.z_size = max([options.z_size, options.pixel_size * 4]);

options_with_precalcs.data.x = [-options.x_size / 2: options.pixel_size: options.x_size / 2] + options.x_offset;
options_with_precalcs.data.z = [0: options.pixel_size: options.z_size] + options.z_offset;
[tmp_mesh.x, tmp_mesh.z] = meshgrid(options_with_precalcs.data.x, options_with_precalcs.data.z);

%shift array postion according to stand off and angle based on either
%specified values or those from finding surface from p/e data
switch options.surface_type
    case '|F|ixed'
        switch options_with_precalcs.x_datum
            case 'Array normal at surf'
                r = options_with_precalcs.array_standoff;
            case 'Array element 1'
                r = (options_with_precalcs.array_standoff - exp_data.array.el_xc(1) * sin(options_with_precalcs.array_inc_angle)) / cos(options_with_precalcs.array_inc_angle);
            case 'Array centre'
                r = options_with_precalcs.array_standoff / cos(options_with_precalcs.array_inc_angle);
        end
        options_with_precalcs.rotation_in_xz_plane = options_with_precalcs.array_inc_angle;
        options_with_precalcs.translation_in_xz_plane = -r * [sin(options_with_precalcs.rotation_in_xz_plane); cos(options_with_precalcs.rotation_in_xz_plane)];
        
        if isfield(options_with_precalcs, 'couplant_result')
            options_with_precalcs = rmfield(options_with_precalcs, 'couplant_result');
        end
    case '|M|easured'
        if options_with_precalcs.s_frac_bandwidth * options_with_precalcs.s_centre_freq > 0
            filter = fn_calc_filter(exp_data.time, options_with_precalcs.s_centre_freq, options_with_precalcs.s_frac_bandwidth * options_with_precalcs.s_centre_freq / 2);
        else
            filter = [];
        end
        [r, n_hat] = fn_find_flat_surf_from_pe_data(exp_data, filter, options_with_precalcs.couplant_velocity, options_with_precalcs.min_dist, options_with_precalcs.max_dist);
        z_hat = [0,0,1];
        options_with_precalcs.rotation_in_xz_plane = acos(dot(-n_hat, z_hat));
        options_with_precalcs.translation_in_xz_plane = -r * [sin(options_with_precalcs.rotation_in_xz_plane), cos(options_with_precalcs.rotation_in_xz_plane)];
end
exp_data.array = fn_move_array_in_xz_plane(exp_data.array, options_with_precalcs.rotation_in_xz_plane, options_with_precalcs.translation_in_xz_plane);
%surface is always along z = 0 (it is the array that moves)
orig_surface.x = [min(min(min(tmp_mesh.x))), max(max(max(tmp_mesh.x)))];
orig_surface.z = [0, 0];

immersion_options = options;
if ~isfield(options_with_precalcs, 'couplant_result')
    %calculate couplant focal law unless couplant image has already
    %been produced
    tmp = exp_data.ph_velocity;
    exp_data.ph_velocity = options_with_precalcs.couplant_velocity;
    options_with_precalcs.load_kernel = 0;
    options_with_precalcs.couplant_focal_law = fn_calc_tfm_focal_law2(exp_data, tmp_mesh, options_with_precalcs);
    options_with_precalcs.couplant_focal_law.interpolation_method = lower(options.interpolation_method);
    options_with_precalcs.couplant_focal_law.filter_on = 1;
    options_with_precalcs.couplant_focal_law.filter = fn_calc_filter(exp_data.time, options_with_precalcs.centre_freq, options_with_precalcs.centre_freq * options_with_precalcs.frac_half_bandwidth / 2);
    exp_data.ph_velocity = tmp;
end
if ~options_with_precalcs.show_couplant_only
    %calculate the sample focal law unless only showing couplant results
    [options_with_precalcs.sample_focal_law, orig_surface] = fn_calc_immersion_tfm_focal_law2(exp_data, tmp_mesh, orig_surface, immersion_options, options_with_precalcs.use_gpu_if_available);
    options_with_precalcs.sample_focal_law.interpolation_method = lower(options.interpolation_method);
    options_with_precalcs.sample_focal_law.filter_on = options.filter_on;
    options_with_precalcs.sample_focal_law.filter = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_half_bandwidth / 2);
end

%show surface on results
options_with_precalcs.geom.lines(1).x = orig_surface.x;
options_with_precalcs.geom.lines(1).y = zeros(size(orig_surface.x));
options_with_precalcs.geom.lines(1).z = orig_surface.z;
options_with_precalcs.geom.lines(1).style = '-';

%show angle limit lines if angle limiter on
% if options.angle_limit_on
%     dx = max(options_with_precalcs.data.z) * sin(options_with_precalcs.focal_law.angle_limit);
%     options_with_precalcs.geom.lines(1).x = [min(exp_data.array.el_xc), max(exp_data.array.el_xc); ...
%         min(exp_data.array.el_xc) + dx, max(exp_data.array.el_xc) - dx];
%     options_with_precalcs.geom.lines(1).y = zeros(2,2);
%     options_with_precalcs.geom.lines(1).z = [0, 0; ones(1,2) * max(options_with_precalcs.data.z)];
%     options_with_precalcs.geom.lines(1).style = ':';
% end

options_with_precalcs.geom.array = fn_get_array_geom_for_plots(exp_data.array);

%GPU check
if (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0) && (options_with_precalcs.use_gpu_if_available)
    if isfield(options_with_precalcs, 'sample_focal_law') && ~isfield(options_with_precalcs.sample_focal_law, 'thread_size')
        gpu_han=gpuDevice(1);
        options_with_precalcs.sample_focal_law.thread_size=gpu_han.MaxThreadsPerBlock;
    end
end
if (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0) && (options_with_precalcs.use_gpu_if_available)
    if isfield(options_with_precalcs, 'couplant_focal_law') && ~isfield(options_with_precalcs.couplant_focal_law, 'thread_size')
        gpu_han=gpuDevice;
        options_with_precalcs.couplant_focal_law.thread_size=gpu_han.MaxThreadsPerBlock;
    end
end

end

%--------------------------------------------------------------------------

function data = fn_process_using_precalcs(exp_data, options_with_precalcs)
%put the actual imaging calculations here, making use of pre-calculated
%values in the options_with_precalcs fields if required.

%copy output coordinates
data.x = options_with_precalcs.data.x;
data.z = options_with_precalcs.data.z;

%apply z-axis offset
switch options_with_precalcs.x_datum
    case 'Array normal at surf'
        %do nothing, this is how everything is calculated anyway
        dx = 0;
    case 'Array element 1'
        dx = -options_with_precalcs.translation_in_xz_plane(1) - exp_data.array.el_xc(1) * cos(options_with_precalcs.rotation_in_xz_plane);
    case 'Array centre'
        dx = -options_with_precalcs.translation_in_xz_plane(1);
end
data.x = data.x + dx;
for ii = 1:length(options_with_precalcs.geom.lines)
    options_with_precalcs.geom.lines(ii).x = options_with_precalcs.geom.lines(ii).x + dx;
end
options_with_precalcs.geom.array.x = options_with_precalcs.geom.array.x + dx;


% generate sample result
if ~options_with_precalcs.show_couplant_only
    sample_result = fn_fast_DAS3(exp_data, options_with_precalcs.sample_focal_law, options_with_precalcs.use_gpu_if_available);
    if  isfield(options_with_precalcs.sample_focal_law,'thread_size')
        sample_result=(gather(sample_result));
    end
end

%generate couplant image if required (i.e. if surface has not been measured
%as if it has, this has already been generated)
if ~isfield(options_with_precalcs, 'couplant_result') %| (isfield(options_with_precalcs, 'couplant_result') && (all(size(options_with_precalcs.couplant_result) ~= size()))
    options_with_precalcs.couplant_result = fn_fast_DAS3(exp_data, options_with_precalcs.couplant_focal_law, options_with_precalcs.use_gpu_if_available);
    if  isfield(options_with_precalcs.couplant_focal_law,'thread_size')
        options_with_precalcs.couplant_result = gather(options_with_precalcs.couplant_result);
    end
end

if ~options_with_precalcs.show_couplant_only
    %merge couplant and sample images
    sample_pts = sum(options_with_precalcs.sample_focal_law.lookup_amp, 3) > 0;
    sample_result = sample_result .* sample_pts;
    couplant_result = options_with_precalcs.couplant_result .* (1 - sample_pts);
    couplant_result = couplant_result / max(max(max(abs(couplant_result)))) * max(max(max(abs(sample_result))));
    data.f = sample_result + couplant_result;
else
    data.f = options_with_precalcs.couplant_result;
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
if isempty(exp_data)
    varargout{1} = [];
    varargout{2} = info;
    return %this is the exit point if exp_data does not exist
end

im_sz_z = max(exp_data.time) * exp_data.ph_velocity / 2;
im_sz_xy = max([max(exp_data.array.el_xc) - min(exp_data.array.el_xc), ...
    max(exp_data.array.el_yc) - min(exp_data.array.el_yc)]);

info.options_info.x_datum.label = 'X Datum';
info.options_info.x_datum.default = 'Array normal at surf';
info.options_info.x_datum.type = 'constrained';
info.options_info.x_datum.constraint = {'Array normal at surf', 'Array element 1', 'Array centre'};

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
    
    info.options_info.y_offset.label = 'X offset (mm)';
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
info.options_info.z_offset.default = 0; %this should be calculated!
info.options_info.z_offset.type = 'double';
info.options_info.z_offset.constraint = [-10, 10];
info.options_info.z_offset.multiplier = 1e-3;

info.options_info.pixel_size.label = 'Pixel size (mm)';
info.options_info.pixel_size.default = round(max([im_sz_xy, im_sz_z]) / no_pixels * 1e3*4)/(1e3*4);
info.options_info.pixel_size.type = 'double';
info.options_info.pixel_size.constraint = [1e-6, 1];
info.options_info.pixel_size.multiplier = 1e-3;

info.options_info.show_couplant_only.label = 'Show couplant only';
info.options_info.show_couplant_only.type = 'bool';
info.options_info.show_couplant_only.constraint = {'On', 'Off'};
info.options_info.show_couplant_only.default = 0;

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

info.options_info.angle_limit_on.label = 'Angle limiter';
info.options_info.angle_limit_on.type = 'bool';
info.options_info.angle_limit_on.constraint = {'On', 'Off'};
info.options_info.angle_limit_on.default = 0;

info.options_info.angle_limit.label = 'Angle limit (degs)';
info.options_info.angle_limit.default = 30 * pi / 180;
info.options_info.angle_limit.type = 'double';
info.options_info.angle_limit.constraint = [1, 89] * pi / 180;
info.options_info.angle_limit.multiplier =  pi / 180;

info.options_info.interpolation_method.label = 'Interpolation';
info.options_info.interpolation_method.default = 'Nearest';
info.options_info.interpolation_method.type = 'constrained';
info.options_info.interpolation_method.constraint = {'Linear', 'Nearest'};

info.options_info.couplant_velocity.label = 'Couplant velocity';
info.options_info.couplant_velocity.default = 1480;
info.options_info.couplant_velocity.type = 'double';
info.options_info.couplant_velocity.constraint = [1,20000];
info.options_info.couplant_velocity.multiplier = 1;

%All about the surface
info.options_info.surface_pts_per_sample_wavelength.label = 'Surface pts/lambda';
info.options_info.surface_pts_per_sample_wavelength.default = 5;
info.options_info.surface_pts_per_sample_wavelength.type = 'double';
info.options_info.surface_pts_per_sample_wavelength.constraint = [0.01, 1000];
info.options_info.surface_pts_per_sample_wavelength.multiplier = 1;

info.options_info.surface_type.label = 'Surface type';
info.options_info.surface_type.default = '|M|easured';
info.options_info.surface_type.type = 'constrained';
info.options_info.surface_type.constraint = {'|F|ixed', '|M|easured'};

%Fixed surface options - none as surface always assumed to start at z = 0
%and align with array (move array instead)

info.options_info.array_standoff.label = '|F| Array standoff (mm)'; %currently based on centre of array
info.options_info.array_standoff.default = im_sz_xy / 2;
info.options_info.array_standoff.type = 'double';
info.options_info.array_standoff.constraint = [1e-6, 1];
info.options_info.array_standoff.multiplier = 1e-3;

info.options_info.array_inc_angle.label = '|F| Array incident angle (degs)';
info.options_info.array_inc_angle.default = 30 * pi / 180;
info.options_info.array_inc_angle.type = 'double';
info.options_info.array_inc_angle.constraint = [-90, 90] * pi / 180;
info.options_info.array_inc_angle.multiplier = pi / 180;

%surface measurement options
info.options_info.min_dist.label = '|M| Min distance (mm)';
if isfield(exp_data.array, 'centre_freq')
    info.options_info.min_dist.default = info.options_info.couplant_velocity.default / exp_data.array.centre_freq * 20;
else
    info.options_info.min_dist.default = 2e-3;
end
info.options_info.min_dist.type = 'double';
info.options_info.min_dist.constraint = [-1, 1];
info.options_info.min_dist.multiplier = 1e-3;

info.options_info.max_dist.label = '|M| Max distance (mm)';
info.options_info.max_dist.default = im_sz_z;
info.options_info.max_dist.type = 'double';
info.options_info.max_dist.constraint = [-1, 1];
info.options_info.max_dist.multiplier = 1e-3;

info.options_info.s_centre_freq.label = '|M| Filter freq (MHz)';
if isfield(exp_data.array, 'centre_freq')
    info.options_info.s_centre_freq.default = exp_data.array.centre_freq / 2;
else
    info.options_info.s_centre_freq.default = 2.5e6;
end;
info.options_info.s_centre_freq.type = 'double';
info.options_info.s_centre_freq.constraint = [0.1,20e6];
info.options_info.s_centre_freq.multiplier = 1e6;

info.options_info.s_frac_bandwidth.label = '|M| Percent b/width';
info.options_info.s_frac_bandwidth.default = 1.5;
info.options_info.s_frac_bandwidth.type = 'double';
info.options_info.s_frac_bandwidth.constraint = [0.01, 10];
info.options_info.s_frac_bandwidth.multiplier = 0.01;

info.options_info.lo_res_pts_per_lambda.label = '|M| Low-res pts/lambda';
info.options_info.lo_res_pts_per_lambda.default = 0.5;
info.options_info.lo_res_pts_per_lambda.type = 'double';
info.options_info.lo_res_pts_per_lambda.constraint = [0.1, 10];
info.options_info.lo_res_pts_per_lambda.multiplier = 1;

info.options_info.hi_res_pts_per_lambda.label = '|M| Hi-res pts/lambda';
info.options_info.hi_res_pts_per_lambda.default = 10;
info.options_info.hi_res_pts_per_lambda.type = 'double';
info.options_info.hi_res_pts_per_lambda.constraint = [0.1, 100];
info.options_info.hi_res_pts_per_lambda.multiplier = 1;

% info.options_info.use_gpu_if_available.label = 'Use GPU if available';
% info.options_info.use_gpu_if_available.type = 'bool';
% info.options_info.use_gpu_if_available.constraint = {'On', 'Off'};
% info.options_info.use_gpu_if_available.default = 1;

end
