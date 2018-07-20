function varargout = fn_contact_comp_wrapper(exp_data, options, mode)
%SUMMARY
%   This picks the basic TFM/SAFT/CSM depending on type of data with
%   extra arguments specifically for transversely isotropic composites
%   (aperture angle limiting, attenuation correction and angle dependent
%   velocity

%USAGE (depending on value of mode argument)
%   initial_info = fn_basic_wrapper([], [], 'return_name_only')
%   extended_info = fn_basic_wrapper(exp_data, [], 'return_info_only')
%   [data, options_with_precalcs] = fn_basic_wrapper(exp_data, options, 'recalc_and_process')
%   data = fn_tfm_wrapper(exp_data, options_with_precalcs, 'process_only')

% default_options.options_changed = 1; %this enables recurring data (e.g. distances to field points to be calculated as a field in options and only recalculated if options_changed = 1)

%the following is the data to allow the processing parameters to be
%displayed and edited in the GUI as well as the default values.
name = 'Composite TFM (contact)'; %name of process that appears on menu
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
if options.angle_limit_on
    calc_focal_law_options.angle_limit = options.angle_limit;
else
    calc_focal_law_options = [];
end
if isfield(options, 'angle_dependent_vel') && options.angle_dependent_vel
    calc_focal_law_options.angle_dep_vel = 1;
end
if data_is_csm
    options_with_precalcs.focal_law = fn_calc_csm_with_ddf_focal_law2(exp_data, tmp_mesh, calc_focal_law_options);
    options_with_precalcs.focal_law.hmc_data = 0;
    if options_with_precalcs.atten_correction_on
        alpha = -log(10 ^ (-options_with_precalcs.atten / 20));
        if ndims(options_with_precalcs.focal_law.lookup_amp_rx) == 3
            zz = options_with_precalcs.data.z(:) * ones(1, size(options_with_precalcs.focal_law.lookup_amp_rx, 2));
            for ii = 1:size(options_with_precalcs.focal_law.lookup_amp_rx, 3)
                options_with_precalcs.focal_law.lookup_amp_rx(:, :, ii) = ...
                    options_with_precalcs.focal_law.lookup_amp_rx(:, :, ii) .* ...
                    exp(alpha * zz);
            end
        else
            zz = shiftdim(repmat(options_with_precalcs.data.z(:), [1, size(options_with_precalcs.focal_law.lookup_amp_rx, 1), size(options_with_precalcs.focal_law.lookup_amp_rx, 2)]), 1);
            for ii = 1:size(options_with_precalcs.focal_law.lookup_amp_rx, 4)
                options_with_precalcs.focal_law.lookup_amp_rx(:, :, :, ii) = ...
                    options_with_precalcs.focal_law.lookup_amp_rx(:, :, :, ii) .* ...
                    exp(alpha * zz);
            end
        end
    end
else
    calc_focal_law_options.interpolation_method = lower(options.interpolation_method);
    options_with_precalcs.focal_law = fn_calc_tfm_focal_law2(exp_data, tmp_mesh, calc_focal_law_options);
    %effect of attenuation correction
    if options_with_precalcs.atten_correction_on
        alpha = -log(10 ^ (-options_with_precalcs.atten / 20));
        if ndims(options_with_precalcs.focal_law.lookup_amp) == 3
            zz = options_with_precalcs.data.z(:) * ones(1, size(options_with_precalcs.focal_law.lookup_amp, 2));
            for ii = 1:size(options_with_precalcs.focal_law.lookup_amp, 3)
                options_with_precalcs.focal_law.lookup_amp(:, :, ii) = ...
                    options_with_precalcs.focal_law.lookup_amp(:, :, ii) .* ...
                    exp(alpha * zz / 2); %over two because correction is applied in Tx and Rx
            end
        else
            zz = shiftdim(repmat(options_with_precalcs.data.z(:), [1, size(options_with_precalcs.focal_law.lookup_amp, 1), size(options_with_precalcs.focal_law.lookup_amp, 2)]), 1);
            for ii = 1:size(options_with_precalcs.focal_law.lookup_amp, 4)
                options_with_precalcs.focal_law.lookup_amp(:, :, :, ii) = ...
                    options_with_precalcs.focal_law.lookup_amp(:, :, :, ii) .* ...
                    exp(alpha * zz / 2); %over two because correction is applied in Tx and Rx
            end
        end
    end
end
options_with_precalcs.focal_law.interpolation_method = lower(options.interpolation_method);
options_with_precalcs.focal_law.filter_on = options.filter_on;
options_with_precalcs.focal_law.filter = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_half_bandwidth / 2);
if options.angle_limit_on
    options_with_precalcs.focal_law.angle_limit = options.angle_limit;
    options_with_precalcs.focal_law.amp_on = 1;
    dx = max(options_with_precalcs.data.z) * tan(options_with_precalcs.focal_law.angle_limit);
    options_with_precalcs.geom.lines(1).x = [min(exp_data.array.el_xc), max(exp_data.array.el_xc); ...
        min(exp_data.array.el_xc) + dx, max(exp_data.array.el_xc) - dx];
    options_with_precalcs.geom.lines(1).y = zeros(2,2);
    options_with_precalcs.geom.lines(1).z = [0, 0; ones(1,2) * max(options_with_precalcs.data.z)];
    options_with_precalcs.geom.lines(1).style = ':';
end
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
if isempty(exp_data)
    return
end

im_sz_z = max(exp_data.time) * exp_data.ph_velocity / 2;
im_sz_xy = max([max(exp_data.array.el_xc) - min(exp_data.array.el_xc), ...
    max(exp_data.array.el_yc) - min(exp_data.array.el_yc)]);
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
    info.options_info.centre_freq.default = 2e6;
end;
info.options_info.centre_freq.type = 'double';
info.options_info.centre_freq.constraint = [0.1, 20e6];
info.options_info.centre_freq.multiplier = 1e6;

info.options_info.frac_half_bandwidth.label = 'Percent b/width';
info.options_info.frac_half_bandwidth.default = 2;
info.options_info.frac_half_bandwidth.type = 'double';
info.options_info.frac_half_bandwidth.constraint = [0.01, 10];
info.options_info.frac_half_bandwidth.multiplier = 0.01;

info.options_info.angle_limit_on.label = 'Angle limiter';
info.options_info.angle_limit_on.type = 'bool';
info.options_info.angle_limit_on.constraint = {'On', 'Off'};
info.options_info.angle_limit_on.default = 1;

if isfield(exp_data, 'vel_poly')
    info.options_info.angle_dependent_vel.label = 'Angle dep. velocity';
    info.options_info.angle_dependent_vel.type = 'bool';
    info.options_info.angle_dependent_vel.constraint = {'On', 'Off'};
    info.options_info.angle_dependent_vel.default = 1;
end

info.options_info.angle_limit.label = 'Angle limit (degs)';
info.options_info.angle_limit.default = 30 * pi / 180;
info.options_info.angle_limit.type = 'double';
info.options_info.angle_limit.constraint = [1, 90] * pi / 180;
info.options_info.angle_limit.multiplier = pi / 180;

info.options_info.atten_correction_on.label = 'Attenuation correction';
info.options_info.atten_correction_on.type = 'bool';
info.options_info.atten_correction_on.constraint = {'On', 'Off'};
info.options_info.atten_correction_on.default = 1;

info.options_info.atten.label = 'Attenuation (dB/mm)';
info.options_info.atten.default = 0;
info.options_info.atten.type = 'double';
info.options_info.atten.constraint = [0, 1e6];
info.options_info.atten.multiplier = 1e3; %ans in dB / m

info.options_info.interpolation_method.label = 'Interpolation';
info.options_info.interpolation_method.default = 'Nearest';
info.options_info.interpolation_method.type = 'constrained';
info.options_info.interpolation_method.constraint = {'Linear', 'Nearest'};

% info.options_info.use_gpu_if_available.label = 'Use GPU if available';
% info.options_info.use_gpu_if_available.type = 'bool';
% info.options_info.use_gpu_if_available.constraint = {'On', 'Off'};
% info.options_info.use_gpu_if_available.default = 1;
end
