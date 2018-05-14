function info = fn_estimate_attenuation_wrapper(exp_data, data, display_options, process_options, h_fn_set_process_options, h_fn_process_options_changed)
if isempty(exp_data) & isempty(data)
    info.name = 'Estimate attenuation';
    return;
else
    info = [];
end;

if ~isfield(display_options, 'select') || size(display_options.select, 1) ~= 2 || size(display_options.select, 2) ~= 2
    warndlg('Select image region first','Warning')
    return;
end

%default size of image to analyse
default_x_min = min(exp_data.array.el_xc);
default_x_max = max(exp_data.array.el_xc);
default_z_min = min(display_options.select(:,2));
default_z_max = max(display_options.select(:,2));

if isfield(process_options, 'angle_limit_on') && process_options.angle_limit_on
    dx = default_z_max * tan(process_options.angle_limit);
    default_x_min = default_x_min + dx;
    default_x_max = default_x_max - dx;
end

%figure size
width = 600;
height = 300;
table_pos = [0,1/2,1/3,1/2];
result_pos = [0,0,1/3,1/2];
graph_pos = [1/3,0,2/3,1];
button_pos = [0,1/4,1/3,1/4];

new_atten_db = 0;

%create figure
p = get(0, 'ScreenSize');
f = figure('Position',[(p(3) - width) / 2, (p(4) - height) / 2, width, height] ,...
    'MenuBar', 'none', ...
    'NumberTitle', 'off', ...
    'ToolBar', 'None', ...
    'Name', ['Analysis:', 'Attenuation estimation'] ...
);

%create graph panel
h_graph_panel = uipanel(f, 'Units', 'Normalized', 'Position', graph_pos);

if isfield(process_options, 'atten')
    %create update button
    h_update_button = uicontrol(f, 'Style', 'pushbutton', 'Units', 'Normalized', 'Position', button_pos, 'String', 'Update process parameter', 'Enable', 'Off', 'Callback', @cb_update);
end

%create options table
[h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(f, table_pos, 'normalized', @fn_new_params);
% keyboard

content_info.x_min.label = 'X min (mm)';
content_info.x_min.default = default_x_min;
content_info.x_min.type = 'double';
content_info.x_min.constraint = [0, 10];
content_info.x_min.multiplier = 1e-3;

content_info.x_max.label = 'X max (mm)';
content_info.x_max.default = default_x_max;
content_info.x_max.type = 'double';
content_info.x_max.constraint = [0, 10];
content_info.x_max.multiplier = 1e-3;

content_info.z_min.label = 'Z min (mm)';
content_info.z_min.default = default_z_min;
content_info.z_min.type = 'double';
content_info.z_min.constraint = [0, 10];
content_info.z_min.multiplier = 1e-3;

content_info.z_max.label = 'Z max (mm)';
content_info.z_max.default = default_z_max;
content_info.z_max.type = 'double';
content_info.z_max.constraint = [0, 10];
content_info.z_max.multiplier = 1e-3;

h_fn_set_content(content_info);

% h_result = uicontrol('Style', 'text', 'Units', 'Normalized', 'Position', result_pos);

a = axes('Parent', h_graph_panel);

%trigger the calc
h_data_changed();

    function fn_new_params(params)
        i1 = min(find(data.x >= params.x_min));
        i2 = max(find(data.x <= params.x_max));
        j1 = min(find(data.z >= params.z_min));
        j2 = max(find(data.z <= params.z_max));
        max_f = max(max(abs(data.f(j1:j2, i1:i2))));
        plot(data.z * 1e3, 20*log10(abs(data.f) / max_f), 'r');
        [atten, atten_db, intercept_db] = fn_calc_atten(data.z(j1:j2), data.f(j1:j2, i1:i2) / max_f);
        hold on;
        plot(data.z * 1e3, intercept_db + atten_db * data.z, 'k:');
        plot(data.z([j1,j2]) * 1e3, intercept_db + atten_db * data.z([j1,j2]), 'k.-');
        title(sprintf('%.2f dB/mm', -atten_db / 1e3));
        new_atten_db = atten_db;
        %enable button so that process option to correct for attenuation
        %can be enabled
        if isfield(process_options, 'atten')
            set(h_update_button, 'Enable', 'On');
        end
    end

    function cb_update(a, b)
        process_options.atten = process_options.atten - new_atten_db;
        h_fn_set_process_options(process_options);
        h_fn_process_options_changed(process_options);
    end
end

function [atten, atten_db, intercept_db] = fn_calc_atten(z, f);
p = zeros(size(f, 2), 2);
for ii = 1:size(f, 2)
    p(ii, :) = polyfit(z(:), 20*log10(abs(f(:, ii))), 1);
end
atten_db = mean(p(:, 1));
intercept_db = mean(p(:, 2));
atten = log(10 ^ (-atten_db / 20));
end