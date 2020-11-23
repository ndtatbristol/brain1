function info = fn_2d_smatrix_wrapper(exp_data, data, options)
if isempty(exp_data) & isempty(data)
    info.name = '2D S-matrix extraction';
    return;
else
    info = [];
end;

if ~isfield(options, 'select') || size(options.select, 1) ~= 2 || size(options.select, 2) ~= 2
    warndlg('Select image region first','Warning')
    return;
end

%params - to be set from window in GUI!
default_pad_factor = 1;
default_ang_pts = 90;
default_density = 2700;
default_speed_ratio = 2;
default_bandwidth = 2e6;
default_centre_freq = 5e6;
default_el_width = 0.53e-3;

% pad_factor = default_pad_factor;
% ang_pts = default_ang_pts;

%figure size
width = 600;
height = 300;
table_pos = [0,1/2,1/3,1/2];
result_pos = [0,0,1/3,1/2];
graph_pos = [1/3,0,2/3,1];

%create figure
p = get(0, 'ScreenSize');
f = figure('Position',[(p(3) - width) / 2, (p(4) - height) / 2, width, height] ,...
    'MenuBar', 'none', ...
    'NumberTitle', 'off', ...
    'ToolBar', 'None', ...
    'Name', ['Analysis:', '2D S-matrix extraction'] ...
    );
%     'WindowStyle', 'Modal', ...

%create graph panel
h_graph_panel = uipanel(f, 'Units', 'Normalized', 'Position', graph_pos);


[h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(f, table_pos, 'normalized', @fn_new_params);
content_info.density.label = 'Density (kg/m^3)';
content_info.density.default = default_density;
content_info.density.type = 'double';
content_info.density.constraint = [1, 20000];
content_info.density.multiplier = 1;

content_info.centre_freq.label = 'Centre frequency (MHz)';
content_info.centre_freq.default = default_centre_freq;
content_info.centre_freq.type = 'double';
content_info.centre_freq.constraint = [1, 1e12];
content_info.centre_freq.multiplier = 1e6;

content_info.bandwidth.label = 'Bandwidth (MHz)';
content_info.bandwidth.default = default_bandwidth;
content_info.bandwidth.type = 'double';
content_info.bandwidth.constraint = [1, 1e12];
content_info.bandwidth.multiplier = 1e6;

content_info.el_width.label = 'Element width (mm)';
if isfield(exp_data.array, 'el_x2') %this is not always defined
    content_info.el_width.default = abs(exp_data.array.el_x2(1) - exp_data.array.el_xc(1)) * 2;
else
    content_info.el_width.default = default_el_width;
end
content_info.el_width.type = 'double';
content_info.el_width.constraint = [1e-5, 1];
content_info.el_width.multiplier = 1e-3;

content_info.speed_ratio.label = 'Long / shear speed ratio';
content_info.speed_ratio.default = default_speed_ratio;
content_info.speed_ratio.type = 'double';
content_info.speed_ratio.constraint = [1, 10];
content_info.speed_ratio.multiplier = 1;

content_info.pad_factor.label = 'Array padding factor';
content_info.pad_factor.default = default_pad_factor;
content_info.pad_factor.type = 'double';
content_info.pad_factor.constraint = [1, 3];
content_info.pad_factor.multiplier = 1;

content_info.ang_pts.label = 'Angular points';
content_info.ang_pts.default = default_ang_pts;
content_info.ang_pts.type = 'int';
content_info.ang_pts.constraint = [36, 3600];
content_info.ang_pts.multiplier = 1;

h_fn_set_content(content_info);

h_result = uicontrol('Style', 'text', 'Units', 'Normalized', 'Position', result_pos);

%various pre-processing of expt data
% exp_data.centre_freq = exp_data.input_freq;
% exp_data.bandwidth = exp_data.input_freq / 4; %er?

a = axes('Parent', h_graph_panel);

%trigger the calc
h_data_changed();

    function fn_new_params(params)
        %determine correct veloicty to use
        if isfield(exp_data, 'vel_elipse') %for legacy files, the spherical harmonic coeffs are not defined for ellipse at this point, so need to read default values from legacy info
            c = exp_data.ph_velocity;
        elseif (isfield(exp_data, 'material') && isfield(exp_data.material, 'vel_spherical_harmonic_coeffs'))
            [c, ~, ~, ~] = fn_get_nominal_velocity(exp_data.material.vel_spherical_harmonic_coeffs);
        elseif isfield(exp_data, 'ph_velocity')
            c = exp_data.ph_velocity;
        else
            error('No valid velocity description found');
        end
        %this is where the actual calculation is called
        exp_data.matl_props.xi = params.speed_ratio;%ratio L to S speed
        exp_data.matl_props.mu = params.density * (c/exp_data.matl_props.xi)^2;
        exp_data.centre_freq = params.centre_freq;
        exp_data.bandwidth = params.bandwidth;
        exp_data.array.el_width = params.el_width;
        exp_data.array.el_pitch = abs(exp_data.array.el_xc(2) - exp_data.array.el_xc(1));
        fn_do_calc(exp_data, options, params.ang_pts, params.pad_factor, h_result);
    end
end

function fn_do_calc(exp_data, options, ang_pts, pad_factor, h_result)
region.x = options.select(:, 1)';
region.z = options.select(:, 2)';
s = fn_1Darray_Smatrix_extraction_v1(exp_data, region, pad_factor);
imagesc(s.phi * 180 / pi, s.phi * 180 / pi,abs(s.m));
axis equal;
axis tight;
colorbar;
xlabel('Incident angle (^o)');
ylabel('Scattering angle (^o)');
end