function options_info = fn_set_standard_fields_for_gui(defaults, varargin)
%SUMMARY
%   Sets up the standard fields for GUI windows about axis sizes and
%   frequency filter
%INPUTS
%   defaults - structure of default axes sizes, pixel size, freq etc. as
%   returned by fn_set_display_options_and_defaults
%   [options_info] - optional options_info to add new fields to. If not
%   specified, then a new variable is created and returned, but this is
%   necessary if you want to have other entries in the list above the ones
%   added by this function
%OUTPUTS
%   options_info - structure with all the necessary fields to populate the
%   UI for axes and filters
%--------------------------------------------------------------------------

if ~isempty(varargin)
    options_info = varargin{1};
end

options_info.x_size.label = 'X size (mm)';
options_info.x_size.default = defaults.im_sz_xy;
options_info.x_size.type = 'double';
options_info.x_size.constraint = [1e-3, 10];
options_info.x_size.multiplier = 1e-3;

options_info.x_offset.label = 'X offset (mm)';
options_info.x_offset.default = 0;
options_info.x_offset.type = 'double';
options_info.x_offset.constraint = [-10, 10];
options_info.x_offset.multiplier = 1e-3;

if defaults.using_2d_array
    options_info.y_size.label = 'Y size (mm)';
    options_info.y_size.default = defaults.im_sz_xy;
    options_info.y_size.type = 'double';
    options_info.y_size.constraint = [1e-3, 10];
    options_info.y_size.multiplier = 1e-3;
    
    options_info.y_offset.label = 'X offset (mm)';
    options_info.y_offset.default = 0;
    options_info.y_offset.type = 'double';
    options_info.y_offset.constraint = [-10, 10];
    options_info.y_offset.multiplier = 1e-3;
end

options_info.z_size.label = 'Z size (mm)';
options_info.z_size.default = defaults.im_sz_z;
options_info.z_size.type = 'double';
options_info.z_size.constraint = [1e-3, 10];
options_info.z_size.multiplier = 1e-3;

options_info.z_offset.label = 'Z offset (mm)';
options_info.z_offset.default = 0; %this should be calculated!
options_info.z_offset.type = 'double';
options_info.z_offset.constraint = [-10, 10];
options_info.z_offset.multiplier = 1e-3;

options_info.pixel_size.label = 'Pixel size (mm)';
options_info.pixel_size.default = defaults.pixel_size;
options_info.pixel_size.type = 'double';
options_info.pixel_size.constraint = [1e-6, 1];
options_info.pixel_size.multiplier = 1e-3;

options_info.show_couplant_only.label = 'Show couplant only';
options_info.show_couplant_only.type = 'bool';
options_info.show_couplant_only.constraint = {'On', 'Off'};
options_info.show_couplant_only.default = 0;

%Filtering

options_info.filter_on.label = 'Filter';
options_info.filter_on.type = 'bool';
options_info.filter_on.constraint = {'On', 'Off'};
options_info.filter_on.default = 1;

options_info.centre_freq.label = 'Filter freq (MHz)';
options_info.centre_freq.default = defaults.freq;
options_info.centre_freq.type = 'double';
options_info.centre_freq.constraint = [0.1, 20e6];
options_info.centre_freq.multiplier = 1e6;

options_info.frac_half_bandwidth.label = 'Percent b/width';
options_info.frac_half_bandwidth.default = defaults.half_bandwidth;
options_info.frac_half_bandwidth.type = 'double';
options_info.frac_half_bandwidth.constraint = [0.01, 10];
options_info.frac_half_bandwidth.multiplier = 0.01;
end