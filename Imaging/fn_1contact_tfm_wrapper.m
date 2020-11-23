function varargout = fn_1contact_tfm_wrapper(exp_data, options, mode)
%SUMMARY
%   This picks the contact TFM/SAFT/CSM depending on type of data for 1D or
%   2D arrays

%USAGE (depending on value of mode argument)
%   initial_info = fn_basic_wrapper([], [], 'return_name_only')
%   extended_info = fn_basic_wrapper(exp_data, [], 'return_info_only')
%   [data, options_with_precalcs] = fn_basic_wrapper(exp_data, options, 'recalc_and_process')
%   data = fn_tfm_wrapper(exp_data, options_with_precalcs, 'process_only')

% default_options.options_changed = 1; %this enables recurring data (e.g. distances to field points to be calculated as a field in options and only recalculated if options_changed = 1)

%the following is the data to allow the processing parameters to be
%displayed and edited in the GUI as well as the default values.

global GPU_PRESENT

name = 'Contact TFM (v3)'; %name of process that appears on menu
name_str=['Imaging: ' name];
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
        plothan=findall(0,'Name',name_str);
        im_han=findall(plothan,'Type','image');
        options_with_precalcs.ax_han=get(im_han,'Parent');
        
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

%Somehow need to hold a copy of the original material (so revert to
%file option works) and have a dummy copy here that is altered by adjusting
%velocity / elliptical velocity profile values

%create original_material field to preserve material originally in exp_data
%so that it can be used if 'From material file' option is selected for velocity
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
% exp_data.ph_velocity = options_with_precalcs.ph_velocity;

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
    options_with_precalcs.data.y=0;
    [tmp_mesh.x, tmp_mesh.z] = meshgrid(options_with_precalcs.data.x, options_with_precalcs.data.z);
end

%calc focal law
if options.angle_limit_on
    if isfield(options, 'look_elev_angle')
        if isfield(options, 'look_azim_angle')
            calc_focal_law_options.angle_limit = [options.angle_limit, options.look_elev_angle, options.look_azim_angle];
        else
            calc_focal_law_options.angle_limit = [options.angle_limit, options.look_elev_angle];
        end
    else
        calc_focal_law_options.angle_limit = options.angle_limit;
    end
    switch options.angle_limit_window
        case 'Hanning'
            calc_focal_law_options.angle_limit_window = 'hanning';
        case 'Rectangular'
            calc_focal_law_options.angle_limit_window = 'rectangular';
    end
    switch options.aperture_weight_correction
        case 'None'
            calc_focal_law_options.aperture_weight_correction = 'none';
        case 'By weights'
            calc_focal_law_options.aperture_weight_correction = 'weights';
        case 'By distance'
            calc_focal_law_options.aperture_weight_correction = 'distance';
    end
else
    calc_focal_law_options = [];
end

%velocity
switch options_with_precalcs.angle_dependent_vel_mode
    case '|E|lliptic'
        lebedev_quality = 3;
        if using_2d_array
            v_y = options_with_precalcs.velocity_y;
        else
            %no y component in options here, so set to same as x
            v_y = options_with_precalcs.velocity_x;
        end
        exp_data.material.vel_spherical_harmonic_coeffs = fn_spherical_harmonics_for_elliptical_profile(options_with_precalcs.velocity_x, v_y, options_with_precalcs.velocity_z, lebedev_quality);
        calc_focal_law_options.angle_dep_vel = 1;
    case 'From material file'
        calc_focal_law_options.angle_dep_vel = 1;
        exp_data.material = options_with_precalcs.original_material;
    otherwise  
        calc_focal_law_options.angle_dep_vel = 0;
        exp_data.material.vel_spherical_harmonic_coeffs = options_with_precalcs.const_velocity;
end

%focal law calculation
if data_is_csm
    calc_focal_law_options.transmit_mode = 'CSM';
end

%actually do the focal law calculation
options_with_precalcs.focal_law = fn_calc_tfm_focal_law3(exp_data, tmp_mesh, calc_focal_law_options);

%do depth attenuation correction if nesc
if options_with_precalcs.atten_correction_on
    if data_is_csm %needs to be applied on rx law only
        alpha = -log(10 ^ (-options_with_precalcs.atten / 20));
        options_with_precalcs.focal_law.lookup_amp_rx = fn_apply_atten(options_with_precalcs.focal_law.lookup_amp_rx, alpha, options_with_precalcs.data.z);
    else
        alpha = -log(10 ^ (-options_with_precalcs.atten / 20)) / 2; %divided by two because it gets applied in tx and rx weightings
        options_with_precalcs.focal_law.lookup_amp = fn_apply_atten(options_with_precalcs.focal_law.lookup_amp, alpha, options_with_precalcs.data.z);
    end
end

options_with_precalcs.focal_law.interpolation_method = lower(options.interpolation_method);
options_with_precalcs.focal_law.filter_on = options.filter_on;
options_with_precalcs.focal_law.filter = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_half_bandwidth / 2);
if options.angle_limit_on
    options_with_precalcs.focal_law.angle_limit = options.angle_limit;
    options_with_precalcs.focal_law.amp_on = 1;
    dx = max(options_with_precalcs.data.z) * sin(options_with_precalcs.focal_law.angle_limit);
    options_with_precalcs.geom.lines(1).x = [min(exp_data.array.el_xc), max(exp_data.array.el_xc); ...
        min(exp_data.array.el_xc) + dx, max(exp_data.array.el_xc) - dx];
    options_with_precalcs.geom.lines(1).y = zeros(2,2);
    options_with_precalcs.geom.lines(1).z = [0, 0; ones(1,2) * max(options_with_precalcs.data.z)];
    options_with_precalcs.geom.lines(1).style = ':';
end
options_with_precalcs.geom.array = fn_get_array_geom_for_plots(exp_data.array);

%GPU check - need to add disable key as well
% if (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0) && (options_with_precalcs.use_gpu_if_available)
%     if ~isfield(options_with_precalcs.focal_law, 'thread_size')
%         options_with_precalcs.focal_law.thread_size=128;
%     end
%     bit_ver=mexext;
%     ptx_file=['gpu_tfm' bit_ver([end-1:end]) '.ptx'];
%     
%     if isfield(options_with_precalcs.focal_law, 'lookup_time')
%         sep_tx_rx_laws = 0;
%     else
%         sep_tx_rx_laws = 1;
%     end
%     
%     if ~isfield(options_with_precalcs.focal_law, 'hmc_data')
%         hmc_data = any(options_with_precalcs.focal_law.tt_weight == 2); %HMC needs to be considered differently if sep tx and rx laws are used
%     else
%         hmc_data = options_with_precalcs.focal_law.hmc_data;
%     end
%     
%     if sep_tx_rx_laws
%         if hmc_data
%             method = 'Different Tx and RX laws, HMC data';
%         else
%             method = 'Different Tx and RX laws, FMC data';
%         end
%     else
%         method = 'Same Tx and RX laws';
%     end
%     
%     method = [method, ' (', options_with_precalcs.focal_law.interpolation_method, ')'];
%     
% %     switch method
% %         case 'Same Tx and RX laws (nearest)'
% %             options_with_precalcs.focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_near_norm');
% %             options_with_precalcs.focal_law.lookup_ind=gpuArray(int32(options_with_precalcs.focal_law.lookup_ind));
% %             options_with_precalcs.focal_law.lookup_amp=gpuArray(single(options_with_precalcs.focal_law.lookup_amp));
% %         case 'Different Tx and RX laws, FMC data (nearest)'
% %             options_with_precalcs.focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_near_2dly');
% %             options_with_precalcs.focal_law.lookup_ind_tx=gpuArray(int32(options_with_precalcs.focal_law.lookup_ind_tx));
% %             options_with_precalcs.focal_law.lookup_ind_rx=gpuArray(int32(options_with_precalcs.focal_law.lookup_ind_rx));
% %             options_with_precalcs.focal_law.lookup_amp_tx=gpuArray(single(options_with_precalcs.focal_law.lookup_amp_tx));
% %             options_with_precalcs.focal_law.lookup_amp_rx=gpuArray(single(options_with_precalcs.focal_law.lookup_amp_rx));
% %         case 'Different Tx and RX laws, HMC data (nearest)'
% %             options_with_precalcs.focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_near_hmc');
% %             options_with_precalcs.focal_law.lookup_ind_tx=gpuArray(int32(options_with_precalcs.focal_law.lookup_ind_tx));
% %             options_with_precalcs.focal_law.lookup_ind_rx=gpuArray(int32(options_with_precalcs.focal_law.lookup_ind_rx));
% %             options_with_precalcs.focal_law.lookup_amp_tx=gpuArray(single(options_with_precalcs.focal_law.lookup_amp_tx));
% %             options_with_precalcs.focal_law.lookup_amp_rx=gpuArray(single(options_with_precalcs.focal_law.lookup_amp_rx));
% %         case 'Same Tx and RX laws (linear)'
% %             options_with_precalcs.focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_norm');
% %             options_with_precalcs.focal_law.lookup_time=gpuArray(single(options_with_precalcs.focal_law.lookup_time));
% %             options_with_precalcs.focal_law.lookup_amp=gpuArray(single(options_with_precalcs.focal_law.lookup_amp));
% %         case 'Different Tx and RX laws, FMC data (linear)'
% %             options_with_precalcs.focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_2dly');
% %             options_with_precalcs.focal_law.lookup_time_tx=gpuArray(single(options_with_precalcs.focal_law.lookup_time_tx));
% %             options_with_precalcs.focal_law.lookup_time_rx=gpuArray(single(options_with_precalcs.focal_law.lookup_time_rx));
% %             options_with_precalcs.focal_law.lookup_amp_tx=gpuArray(single(options_with_precalcs.focal_law.lookup_amp_tx));
% %             options_with_precalcs.focal_law.lookup_amp_rx=gpuArray(single(options_with_precalcs.focal_law.lookup_amp_rx));
% %         case 'Different Tx and RX laws, HMC data (linear)'
% %             options_with_precalcs.focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_hmc');
% %             options_with_precalcs.focal_law.lookup_time_tx=gpuArray(single(options_with_precalcs.focal_law.lookup_time_tx));
% %             options_with_precalcs.focal_law.lookup_time_rx=gpuArray(single(options_with_precalcs.focal_law.lookup_time_rx));
% %             options_with_precalcs.focal_law.lookup_amp_tx=gpuArray(single(options_with_precalcs.focal_law.lookup_amp_tx));
% %             options_with_precalcs.focal_law.lookup_amp_rx=gpuArray(single(options_with_precalcs.focal_law.lookup_amp_rx));
% %     end
% %     
%     options_with_precalcs.focal_law.kern.ThreadBlockSize = options_with_precalcs.focal_law.thread_size;
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
end
data.z = options_with_precalcs.data.z;
%the actual calculation

data.f = fn_fast_DAS3(exp_data, options_with_precalcs.focal_law, use_gpu);
if isfield(options_with_precalcs.focal_law,'thread_size')
   data.f = gather(data.f);
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
end
default_pixel_size = round(default_pixel_size / round_to) * round_to;


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
info.options_info.z_offset.default = 0;
info.options_info.z_offset.type = 'double';
info.options_info.z_offset.constraint = [-10, 10];
info.options_info.z_offset.multiplier = 1e-3;

info.options_info.pixel_size.default = default_pixel_size;
info.options_info.pixel_size.label = 'Pixel size (mm)';
info.options_info.pixel_size.type = 'double';
info.options_info.pixel_size.constraint = [1e-6, 1];
info.options_info.pixel_size.multiplier = 1e-3;

%Filtering

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

%Constant velocity

info.options_info.const_velocity.label = 'Velocity (m/s)';
info.options_info.const_velocity.default = v;
info.options_info.const_velocity.type = 'double';
info.options_info.const_velocity.constraint = [1, 20000];
info.options_info.const_velocity.multiplier = 1;

%Control for angle dependent velocity and options to set ellipsoidal vel

info.options_info.angle_dependent_vel_mode.label = 'Angle dep. velocity';
info.options_info.angle_dependent_vel_mode.type = 'constrained';
info.options_info.angle_dependent_vel_mode.constraint = {'None', '|E|lliptic', 'From material file'};
info.options_info.angle_dependent_vel_mode.default = 'None';

info.options_info.velocity_x.label = '|E| x-velocity (m/s)';
info.options_info.velocity_x.default = v_x;
info.options_info.velocity_x.type = 'double';
info.options_info.velocity_x.constraint = [1, 20000];
info.options_info.velocity_x.multiplier = 1;

if using_2d_array
    info.options_info.velocity_y.label = '|E| y-velocity (m/s)';
    info.options_info.velocity_y.default = v_y;
    info.options_info.velocity_y.type = 'double';
    info.options_info.velocity_y.constraint = [1, 20000];
    info.options_info.velocity_y.multiplier = 1;
end

info.options_info.velocity_z.label = '|E| z-velocity (m/s)';
info.options_info.velocity_z.default = v_z;
info.options_info.velocity_z.type = 'double';
info.options_info.velocity_z.constraint = [1, 20000];
info.options_info.velocity_z.multiplier = 1;




% if isfield(exp_data, 'vel_poly') || isfield(exp_data, 'vel_elipse')
%    if isfield(exp_data, 'vel_elipse')
%         info.options_info.angle_dependent_vel.default = 'Elliptic';
%         info.options_info.ellipse_vel_x.default = exp_data.vel_elipse(1);
%         info.options_info.ellipse_vel_z.default = exp_data.vel_elipse(2);
%     else
%         %this is just for legacy files - no option to change values from
%         %imaging window
%         info.options_info.angle_dependent_vel.default = 'Polynomial';
%         info.options_info.ellipse_vel_x.default = exp_data.ph_velocity;
%         info.options_info.ellipse_vel_z.default = exp_data.ph_velocity;
%     end
% else
%     info.options_info.angle_dependent_vel.default = 'None';
%     info.options_info.ellipse_vel_x.default = exp_data.ph_velocity;
%     info.options_info.ellipse_vel_z.default = exp_data.ph_velocity;
% end

info.options_info.atten_correction_on.label = 'Attenuation correction';
info.options_info.atten_correction_on.type = 'bool';
info.options_info.atten_correction_on.constraint = {'On', 'Off'};
info.options_info.atten_correction_on.default = 1;

info.options_info.atten.label = 'Attenuation (dB/mm)';
info.options_info.atten.default = 0;
info.options_info.atten.type = 'double';
info.options_info.atten.constraint = [0, 1e6];
info.options_info.atten.multiplier = 1e3; %ans in dB / m

%Aperture control
info.options_info.angle_limit_on.label = 'Angle limiter';
info.options_info.angle_limit_on.type = 'bool';
info.options_info.angle_limit_on.constraint = {'On', 'Off'};
info.options_info.angle_limit_on.default = 0;

info.options_info.angle_limit.label = 'Angle limit (degs)';
info.options_info.angle_limit.default = 30 * pi / 180;
info.options_info.angle_limit.type = 'double';
info.options_info.angle_limit.constraint = [1, 90] * pi / 180;
info.options_info.angle_limit.multiplier = pi / 180;

info.options_info.look_elev_angle.label = 'Look elevation angle (degs)';
info.options_info.look_elev_angle.default = 0 * pi / 180;
info.options_info.look_elev_angle.type = 'double';
info.options_info.look_elev_angle.constraint = [-90, 90] * pi / 180;
info.options_info.look_elev_angle.multiplier = pi / 180;

if using_2d_array %need azimuthal angle as well
    info.options_info.look_azim_angle.label = 'Look azimuthal angle (degs)';
    info.options_info.look_azim_angle.default = 0 * pi / 180;
    info.options_info.look_azim_angle.type = 'double';
    info.options_info.look_azim_angle.constraint = [-180, 180] * pi / 180;
    info.options_info.look_azim_angle.multiplier = pi / 180;
end
    
info.options_info.angle_limit_window.label = 'Aperture window';
info.options_info.angle_limit_window.default = 'Hanning';
info.options_info.angle_limit_window.type = 'constrained';
info.options_info.angle_limit_window.constraint = {'Hanning', 'Rectangular'};

info.options_info.aperture_weight_correction.label = 'Aperture weight correction';
info.options_info.aperture_weight_correction.default = 'By weights';
info.options_info.aperture_weight_correction.type = 'constrained';
info.options_info.aperture_weight_correction.constraint = {'None', 'By weights', 'By distance'};

%DAS controls
info.options_info.interpolation_method.label = 'Interpolation';
info.options_info.interpolation_method.default = 'Linear';
info.options_info.interpolation_method.type = 'constrained';
info.options_info.interpolation_method.constraint = {'Linear', 'Nearest'};

%Note: option to switch GPU on/off is automatically added by gui_process_window
%if GPU is present. No need to add it here.

end

function out_amp = fn_apply_atten(in_amp, alpha, z)
if ndims(in_amp) == 3
    atten = repmat(exp(alpha * z(:)), [1, size(in_amp, 2), size(in_amp, 3)]);
else
    atten = permute(repmat(exp(alpha * z(:)), [1, size(in_amp, 1), size(in_amp, 2), size(in_amp, 4)]), [2,3,1,4]);
end
out_amp = in_amp .* atten;
end