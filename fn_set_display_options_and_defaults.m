function [display_options, defaults, fn_display] = fn_set_display_options_and_defaults(exp_data, gpu_present)
%SUMMARY
%   Calculates various default values for displays based on form of
%   experimental data (2D/3D, freq of array, number of time points) as well
%   as setting up the appropriate axis options and display function to use.
%INPUTS
%   exp_data - experimental data structure
%   gpu_present - is GPU present and working flag (1 or 0)
%OUTPUTS
%   display_options - structure containin the display options (axis scalings etc)
%   defaults - structure containing default axes lengths, pixel sizes
%   fn_display - pointer to appropriate display function
%--------------------------------------------------------------------------
config = fn_get_config;

%check style of data and array
[defaults.using_2d_array, ~] = fn_check_form_of_exp_data(exp_data);

%Nominal velocities
[defaults.v, defaults.v_x, defaults.v_y, defaults.v_z] = fn_get_nominal_velocity(exp_data.material.vel_spherical_harmonic_coeffs);

%Frequency and bandwidth
if isfield(exp_data.array, 'centre_freq') %PW -used in various places below to set initial values
    defaults.freq = exp_data.array.centre_freq;
else
    defaults.freq = config.default_centre_freq;
end
defaults.half_bandwidth = config.default_fractional_bandwidth; %note name of this is misleading - it is full fractional bandwidth! Some sort of legacy issue

%Basic axis settings
display_options.axis_equal = 1;
display_options.x_axis_sf = 1e3;
display_options.z_axis_sf = 1e3;

%Default lengths of axes
defaults.im_sz_z = max(exp_data.time) * defaults.v / 2;
defaults.im_sz_xy = max([...
    max(exp_data.array.el_xc) - min(exp_data.array.el_xc), ...
    max(exp_data.array.el_yc) - min(exp_data.array.el_yc)]);

%Default size of pixels set based on presence of GPU and dimensionality of data
if defaults.using_2d_array
    fn_display = @gui_3d_plot_panel;
    display_options.interpolation = 0;
    display_options.y_axis_sf = 1e3;
    no_pixels = config.default_size_of_3d_image_in_pixels;
    defaults.pixel_size = max([defaults.im_sz_xy, defaults.im_sz_z]) / no_pixels;
    round_to = 0.25e-3;
else
    fn_display = @gui_2d_plot_panel;
    display_options.interpolate = 0;
    if gpu_present 
        max_freq = defaults.freq * (1 + defaults.half_bandwidth / 2);
        defaults.pixel_size= (defaults.v ./ max_freq) / 4; %divide by four is to account for nyquist frequency and out and back path length;
        round_to = 0.01e-3;
    else
        no_pixels = config.default_size_of_2d_image_in_pixels;
        defaults.pixel_size = max([defaults.im_sz_xy, defaults.im_sz_z]) / no_pixels;
        round_to = 0.25e-3;
    end
end
defaults.pixel_size = round(defaults.pixel_size / round_to) * round_to;

if gpu_present
    info.display_options.gpu = 1;
else
	info.display_options.gpu = 0;
end


end