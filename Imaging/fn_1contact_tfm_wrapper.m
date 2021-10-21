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
        %plothan=findall(0,'Name',name_str);
        %im_han=findall(plothan,'Type','image');
        %options_with_precalcs.ax_han=get(im_han,'Parent');
        
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

%check style of data and array
[using_2d_array, data_is_csm] = fn_check_form_of_exp_data(exp_data);

%set up grid and image axes
[options_with_precalcs.data, tmp_mesh] = fn_set_up_image_mesh(options, using_2d_array);

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

%Set load kernel option if GPU is present to increase refresh rate 
%(otherwise kernel is loaded to GPU separately for each frame)
if GPU_PRESENT
	calc_focal_law_options.load_kernel = 1;
end

%Do focal law calulation
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

%Get default values for image etc and set display options
[info.display_options, defaults, info.fn_display] = fn_set_display_options_and_defaults(exp_data, GPU_PRESENT);

%--------------------------------------------------------------------------
%Populate the controls

%Fill the basic ones about image size and filter
info.options_info = fn_set_standard_fields_for_gui(defaults);

%Constant velocity
info.options_info.const_velocity.label = 'Velocity (m/s)';
info.options_info.const_velocity.default = defaults.v;
info.options_info.const_velocity.type = 'double';
info.options_info.const_velocity.constraint = [1, 20000];
info.options_info.const_velocity.multiplier = 1;

%Control for angle dependent velocity and options to set ellipsoidal vel
info.options_info.angle_dependent_vel_mode.label = 'Angle dep. velocity';
info.options_info.angle_dependent_vel_mode.type = 'constrained';
info.options_info.angle_dependent_vel_mode.constraint = {'None', '|E|lliptic', 'From material file'};
info.options_info.angle_dependent_vel_mode.default = 'None';

info.options_info.velocity_x.label = '|E| x-velocity (m/s)';
info.options_info.velocity_x.default = defaults.v_x;
info.options_info.velocity_x.type = 'double';
info.options_info.velocity_x.constraint = [1, 20000];
info.options_info.velocity_x.multiplier = 1;

if defaults.using_2d_array
    info.options_info.velocity_y.label = '|E| y-velocity (m/s)';
    info.options_info.velocity_y.default = defaults.v_y;
    info.options_info.velocity_y.type = 'double';
    info.options_info.velocity_y.constraint = [1, 20000];
    info.options_info.velocity_y.multiplier = 1;
end

info.options_info.velocity_z.label = '|E| z-velocity (m/s)';
info.options_info.velocity_z.default = defaults.v_z;
info.options_info.velocity_z.type = 'double';
info.options_info.velocity_z.constraint = [1, 20000];
info.options_info.velocity_z.multiplier = 1;

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

if defaults.using_2d_array %need azimuthal angle as well
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