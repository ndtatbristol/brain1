function varargout = fn_10ascan_wrapper(exp_data, options, mode)
%SUMMARY
%   Simple A-scans for groups of elements with steering only (1D arrays
%   only at present)
%USAGE (depending on value of mode argument)
%   initial_info = fn_basic_wrapper([], [], 'return_name_only')
%   extended_info = fn_basic_wrapper(exp_data, [], 'return_info_only')
%   [data, options_with_precalcs] = fn_basic_wrapper(exp_data, options, 'recalc_and_process')
%   data = fn_tfm_wrapper(exp_data, options_with_precalcs, 'process_only')

% default_options.options_changed = 1; %this enables recurring data (e.g. distances to field points to be calculated as a field in options and only recalculated if options_changed = 1)

%the following is the data to allow the processing parameters to be
%displayed and edited in the GUI as well as the default values.
name = 'Simple A-scans'; %name of process that appears on menu
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

options_with_precalcs.data.x = exp_data.time;
options_with_precalcs.data.z = [1:3];%up to 3 A-scans will be shown

using_2d_array = any(exp_data.array.el_yc);

if using_2d_array
    r1 = sqrt((exp_data.array.el_xc - options.centx1) .^ 2 + (exp_data.array.el_yc - options.centy1) .^ 2);
    r2 = sqrt((exp_data.array.el_xc - options.centx2) .^ 2 + (exp_data.array.el_yc - options.centy2) .^ 2);
    r3 = sqrt((exp_data.array.el_xc - options.centx3) .^ 2 + (exp_data.array.el_yc - options.centy3) .^ 2);
else
    r1 = abs(exp_data.array.el_xc - options.centx1);
    r2 = abs(exp_data.array.el_xc - options.centx2);
    r3 = abs(exp_data.array.el_xc - options.centx3);
end
els_1 = find(r1 < (options.size1 / 2));
els_2 = find(r2 < (options.size2 / 2));
els_3 = find(r3 < (options.size3 / 2));
    
%calc focal law -  different if data is CSM!
data_is_csm = length(unique(exp_data.tx)) == 1;
if data_is_csm
    options_with_precalcs.tt_indices1 = find(ismember(exp_data.rx, els_1));
    options_with_precalcs.tt_indices2 = find(ismember(exp_data.rx, els_2));
    options_with_precalcs.tt_indices3 = find(ismember(exp_data.rx, els_3));
    
else
    options_with_precalcs.tt_indices1 = find(ismember(exp_data.tx, els_1) & ismember(exp_data.rx, els_1));
    options_with_precalcs.tt_indices2 = find(ismember(exp_data.tx, els_2) & ismember(exp_data.rx, els_2));
    options_with_precalcs.tt_indices3 = find(ismember(exp_data.tx, els_3) & ismember(exp_data.rx, els_3));
end

options_with_precalcs.focal_law.filter_on = options.filter_on;
options_with_precalcs.focal_law.filter = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_half_bandwidth / 2);
options_with_precalcs.focal_law.hilbert_on = 1;
options_with_precalcs.geom.array = fn_get_array_geom_for_plots(exp_data.array);

end

%--------------------------------------------------------------------------

function data = fn_process_using_precalcs(exp_data, options_with_precalcs)
%put the actual imaging calculations here, making use of pre-calculated
%values in the options_with_precalcs fields if required.

%copy output output coordinates
data.x = options_with_precalcs.data.x;
if isfield(options_with_precalcs.data, 'y')
    data.y = options_with_precalcs.data.y;
end
data.z = options_with_precalcs.data.z;

%the actual calculation
data.f = zeros(length(data.x), length(data.z));
data.f(:,1) = sum(exp_data.time_data(:,options_with_precalcs.tt_indices1), 2);
data.f(:,2) = sum(exp_data.time_data(:,options_with_precalcs.tt_indices2), 2);
data.f(:,3) = sum(exp_data.time_data(:,options_with_precalcs.tt_indices3), 2);

%filter
if options_with_precalcs.focal_law.filter_on
    data.f = ifft(spdiags(options_with_precalcs.focal_law.filter, 0, length(exp_data.time), length(exp_data.time)) * fft(data.f));
else
    if options_with_precalcs.focal_law.hilbert_on
        data.f = ifft(spdiags([1:length(exp_data.time)]' < length(exp_data.time) / 2, 0, length(exp_data.time), length(exp_data.time)) * fft(data.f));
    else
        %do nothing!
    end
end
data.f = data.f .'; %because of convention used elsewhere!
data.geom = options_with_precalcs.geom;
end

%--------------------------------------------------------------------------

function info = fn_return_info(exp_data)

using_2d_array = ~isempty(exp_data) && any(exp_data.array.el_yc);

info.fn_display = @gui_1d_plot_panel;
info.display_options.interpolation = 0;
info.display_options.axis_equal = 0;
info.display_options.x_axis_sf = 1e6;
info.display_options.y_axis_sf = 1;
info.display_options.z_axis_sf = 1;
if isempty(exp_data)
    varargout{1} = [];
    varargout{2} = info;
    return %this is the exit point if exp_data does not exist
end

default_step = 1;
default_aperture = 10;
% 
if using_2d_array
    xc = mean(exp_data.array.el_xc(:));
    yc = mean(exp_data.array.el_yc(:));
    r = ((max(exp_data.array.el_xc(:)) - min(exp_data.array.el_xc(:))) + (max(exp_data.array.el_yc(:)) - min(exp_data.array.el_yc(:)))) / 4; %mean semi-axis length
    a = linspace(0,2*pi,4); %divide into 3 round apertures 
    x1 = r / 2 * cos(a(1));
    y1 = r / 2 * sin(a(1));
    x2 = r / 2 * cos(a(2));
    y2 = r / 2 * sin(a(2));
    x3 = r / 2 * cos(a(3));
    y3 = r / 2 * sin(a(3));
    d = r;
else
    xc = mean(exp_data.array.el_xc(:));
    d = (max(exp_data.array.el_xc(:)) - min(exp_data.array.el_xc(:))) / 3; %divide aperture into 3 blocks
    x1 = xc - d;
    x2 = xc;
    x3 = xc + d;
end

%1st A-scan
info.options_info.centx1.label = 'A-scan 1 apert. cent. X (mm)';
info.options_info.centx1.default = x1;
info.options_info.centx1.type = 'double';
info.options_info.centx1.constraint = [min(exp_data.array.el_xc), max(exp_data.array.el_xc)];
info.options_info.centx1.multiplier = 1e-3;

if using_2d_array
    info.options_info.centy1.label = 'A-scan 1 apert. cent. Y (mm)';
    info.options_info.centy1.default = y1;
    info.options_info.centy1.type = 'double';
    info.options_info.centy1.constraint = [min(exp_data.array.el_yc), max(exp_data.array.el_yc)];
    info.options_info.centy1.multiplier = 1e-3;
end

info.options_info.size1.label = 'A-scan 1 apert. size (mm)';
info.options_info.size1.default = d;
info.options_info.size1.type = 'double';
info.options_info.size1.constraint = [0, max(exp_data.array.el_xc) - min(exp_data.array.el_xc)];
info.options_info.size1.multiplier = 1e-3;

%2nd A-scan
info.options_info.centx2.label = 'A-scan 2 apert. cent. X (mm)';
info.options_info.centx2.default = x2;
info.options_info.centx2.type = 'double';
info.options_info.centx2.constraint = [min(exp_data.array.el_xc), max(exp_data.array.el_xc)];
info.options_info.centx2.multiplier = 1e-3;

if using_2d_array
    info.options_info.centy2.label = 'A-scan 2 apert. cent. Y (mm)';
    info.options_info.centy2.default = y2;
    info.options_info.centy2.type = 'double';
    info.options_info.centy2.constraint = [min(exp_data.array.el_yc), max(exp_data.array.el_yc)];
    info.options_info.centy2.multiplier = 1e-3;
end

info.options_info.size2.label = 'A-scan 2 apert. size (mm)';
info.options_info.size2.default = d;
info.options_info.size2.type = 'double';
info.options_info.size2.constraint = [0, max(exp_data.array.el_xc) - min(exp_data.array.el_xc)];
info.options_info.size2.multiplier = 1e-3;

%3rd A-scan
info.options_info.centx3.label = 'A-scan 3 apert. cent. X (mm)';
info.options_info.centx3.default = x3;
info.options_info.centx3.type = 'double';
info.options_info.centx3.constraint = [min(exp_data.array.el_xc), max(exp_data.array.el_xc)];
info.options_info.centx3.multiplier = 1e-3;

if using_2d_array
    info.options_info.centy3.label = 'A-scan 3 apert. cent. Y (mm)';
    info.options_info.centy3.default = y3;
    info.options_info.centy3.type = 'double';
    info.options_info.centy3.constraint = [min(exp_data.array.el_yc), max(exp_data.array.el_yc)];
    info.options_info.centy3.multiplier = 1e-3;
end

info.options_info.size3.label = 'A-scan 3 apert. size (mm)';
info.options_info.size3.default = d;
info.options_info.size3.type = 'double';
info.options_info.size3.constraint = [0, max(exp_data.array.el_xc) - min(exp_data.array.el_xc)];
info.options_info.size3.multiplier = 1e-3;



% info.options_info.el_max1.label = 'A-scan 1 last element';
% info.options_info.el_max1.default = default_aperture;
% info.options_info.el_max1.type = 'int';
% info.options_info.el_max1.constraint = [1, length(exp_data.array.el_xc)];

%2nd A-scan
% info.options_info.el_min2.label = 'A-scan 2 first element';
% info.options_info.el_min2.default = round((length(exp_data.array.el_xc) - default_aperture) / 2);
% info.options_info.el_min2.type = 'int';
% info.options_info.el_min2.constraint = [1, length(exp_data.array.el_xc)];
% 
% info.options_info.el_step2.label = 'A-scan 2 element step';
% info.options_info.el_step2.default = default_step;
% info.options_info.el_step2.type = 'int';
% info.options_info.el_step2.constraint = [1, length(exp_data.array.el_xc)];
% 
% info.options_info.el_max2.label = 'A-scan 2 last element';
% info.options_info.el_max2.default = round((length(exp_data.array.el_xc) - default_aperture) / 2) + default_aperture - 1;
% info.options_info.el_max2.type = 'int';
% info.options_info.el_max2.constraint = [1, length(exp_data.array.el_xc)];
% 
% %3rd A-scan
% info.options_info.el_min3.label = 'A-scan 3 first element';
% info.options_info.el_min3.default = length(exp_data.array.el_xc) - default_aperture + 1;
% info.options_info.el_min3.type = 'int';
% info.options_info.el_min3.constraint = [1, length(exp_data.array.el_xc)];
% 
% info.options_info.el_step3.label = 'A-scan 3 element step';
% info.options_info.el_step3.default = default_step;
% info.options_info.el_step3.type = 'int';
% info.options_info.el_step3.constraint = [1, length(exp_data.array.el_xc)];
% 
% info.options_info.el_max3.label = 'A-scan 3 last element';
% info.options_info.el_max3.default = length(exp_data.array.el_xc);
% info.options_info.el_max3.type = 'int';
% info.options_info.el_max3.constraint = [1, length(exp_data.array.el_xc)];

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

% info.options_info.ph_velocity.label = 'Velocity (m/s)';
% info.options_info.ph_velocity.default = exp_data.ph_velocity;
% info.options_info.ph_velocity.type = 'double';
% info.options_info.ph_velocity.constraint = [1, 20000];
% info.options_info.ph_velocity.multiplier = 1;

end
