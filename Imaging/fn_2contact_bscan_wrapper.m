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
global GPU_PRESENT

options_with_precalcs = options; %need this line as initial options 

%check style of data and array
[using_2d_array, data_is_csm] = fn_check_form_of_exp_data(exp_data);

%set up grid and image axes
[options_with_precalcs.data, tmp_mesh] = fn_set_up_image_mesh(options, using_2d_array);

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

%GPU check - need to add disable key as well
if GPU_PRESENT
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
if use_gpu
   data.f = gather(data.f);
end
data.geom = options_with_precalcs.geom;
end

%--------------------------------------------------------------------------

function info = fn_return_info(exp_data)

global GPU_PRESENT

%Get default values for image etc and set display options
[info.display_options, defaults, info.fn_display] = fn_set_display_options_and_defaults(exp_data, GPU_PRESENT);

array_size = mean([max(exp_data.array.el_xc(:)) - min(exp_data.array.el_xc(:)), max(exp_data.array.el_yc(:)) - min(exp_data.array.el_yc(:))]);
defaults.aperture_size = array_size / 8;

%--------------------------------------------------------------------------
%Populate the controls

%Fill the basic ones about image size and filter
info.options_info = fn_set_standard_fields_for_gui(defaults);

info.options_info.const_velocity.label = 'Velocity (m/s)';
info.options_info.const_velocity.default = defaults.v;
info.options_info.const_velocity.type = 'double';
info.options_info.const_velocity.constraint = [1, 20000];
info.options_info.const_velocity.multiplier = 1;

info.options_info.aperture_size_metres.label = 'Aperture size (mm)';
info.options_info.aperture_size_metres.default = defaults.aperture_size;
info.options_info.aperture_size_metres.type = 'double';
info.options_info.aperture_size_metres.constraint = [0.0001, 10];
info.options_info.aperture_size_metres.multiplier = 1e-3;

info.options_info.interpolation_method.label = 'Interpolation';
info.options_info.interpolation_method.default = 'Nearest';
info.options_info.interpolation_method.type = 'constrained';
info.options_info.interpolation_method.constraint = {'Linear', 'Nearest'};

%Note: option to switch GPU on/off is automatically added by gui_process_window
%if GPU is present. No need to add it here.

end
