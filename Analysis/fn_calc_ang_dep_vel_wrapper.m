function info = fn_calc_ang_dep_vel_wrapper(exp_data, data, display_options, process_options, h_fn_set_process_options, h_fn_process_options_changed)

%Calculates vel profile for contact array on transversely isotropic sample
%assuming parallel flat back wall.

%Requires an image generated using a nominal velocity first and the region
%containing the BWE to be selected. Sample thickness will be estimated
%based on arrival time of normal incidence A-scan peak in this window using
%nominal phase velocity. Thickness value can be overwritten from this window if necessary.
%Writing new material file will use ph_velocity (recalculated if thickness
%altered) and vel_poly structure

%load configuration data
config = fn_get_config;

if isempty(exp_data) & isempty(data)
    info.name = 'Calculate angle dep. velocity';
    return;
else
    info = [];
end;

if ~isfield(display_options, 'select') || size(display_options.select, 1) ~= 2 || size(display_options.select, 2) ~= 2
    warndlg('Select image region first','Warning')
    return;
end

%filter exp_data a/c to current settings
dt = exp_data.time(2) - exp_data.time(1);
if process_options.filter_on
    filter = fn_gaussian(length(exp_data.time), process_options.centre_freq * dt, process_options.frac_half_bandwidth * process_options.centre_freq * dt / 2);
    exp_data.time_data = ifft(spdiags(filter, 0, length(exp_data.time), length(exp_data.time)) * fft(exp_data.time_data));
end

%work out arrival time of BWE by simple sum across all data
ascan = abs(fn_hilbert(sum(exp_data.time_data, 2)));
z = exp_data.time * exp_data.ph_velocity / 2;

z_min = min(display_options.select(:,2));
z_max = max(display_options.select(:,2));
[dummy, jj] = max(ascan .* (z > z_min) .* (z < z_max));
default_thickness = z(jj);

default_min_std_over_mean = 0.8;
default_max_fract_vel_shift = 0.1;
default_poly_order = 4;
default_force_even_poly = 1;
default_name = 'New material';

material.name = sprintf([default_name, ' (%i)'], round(exp_data.ph_velocity));
material.ph_velocity = exp_data.ph_velocity;

% fname = fullfile(start_in_folder, config.files.local_brain_path, config.files.materials_path, material.name);

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

%create update file button
h_update_button = uicontrol(f, 'Style', 'pushbutton', 'Units', 'Normalized', 'Position', button_pos, 'String', 'Save new material', 'Enable', 'Off', 'Callback', @cb_update);

%create options table
[h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(f, table_pos, 'normalized', @fn_new_params);

content_info.name.label = 'Name';
content_info.name.default = default_name;
content_info.name.type = 'string';

content_info.thickness.label = 'Thickness (mm)';
content_info.thickness.default = default_thickness;
content_info.thickness.type = 'double';
content_info.thickness.constraint = [0, 10];
content_info.thickness.multiplier = 1e-3;

content_info.min_std_over_mean.label = 'Min std / mean';
content_info.min_std_over_mean.default = default_min_std_over_mean;
content_info.min_std_over_mean.type = 'double';
content_info.min_std_over_mean.constraint = [0, 5];
content_info.min_std_over_mean.multiplier = 1;

content_info.max_fract_vel_shift.label = 'Max vel. shift (%)';
content_info.max_fract_vel_shift.default = default_max_fract_vel_shift;
content_info.max_fract_vel_shift.type = 'double';
content_info.max_fract_vel_shift.constraint = [0, 100];
content_info.max_fract_vel_shift.multiplier = 0.01;

content_info.poly_order.label = 'Poly order';
content_info.poly_order.default = default_poly_order;
content_info.poly_order.type = 'int';
content_info.poly_order.constraint = [2, 6];

content_info.force_even_poly.label = 'Force even poly';
content_info.force_even_poly.type = 'bool';
content_info.force_even_poly.constraint = {'Yes', 'No'};
content_info.force_even_poly.default = default_force_even_poly;


h_fn_set_content(content_info);

a = axes('Parent', h_graph_panel);

%trigger the calc
h_data_changed();

    function fn_new_params(params)
        %the actual calculation
        [ph_velocity, vel_poly, time_data2, theta, arrival_time, dist] = fn_calc_vel_poly(exp_data, params.thickness, z_min, z_max, params.max_fract_vel_shift, params.min_std_over_mean, params.poly_order, params.force_even_poly);

        %update material details and name
        material.ph_velocity = ph_velocity;
        material.vel_poly = vel_poly;
        material.name = sprintf([default_name, ' (%i var)'], round(exp_data.ph_velocity));
%         fname = fullfile(start_in_folder, config.files.local_brain_path, config.files.materials_path, material.name);
        
        %display results
        subplot(1,2,1);
        cla;
        imagesc(theta * 180 / pi, exp_data.time * 1e6, log(abs(time_data2)));
        hold on;
        kk = find(arrival_time > 0);
        plot(theta(kk) * 180 / pi, arrival_time(kk) * 1e6, 'k.');
        xlabel('Angle (^o)');
        ylabel('Time (us)');
        zoom on;

        subplot(1,2,2);
        cla;
        arrival_time(find(arrival_time == 0)) = NaN;
        plot(theta * 180 / pi, dist ./ arrival_time, 'k.');
        vel = polyval(vel_poly.p, theta, [], vel_poly.mu);
        hold on;
        plot(theta * 180 / pi, vel, 'r');
        xlabel('Angle (^o)');
        ylabel('Velocity (m/s)');
        zoom on;

        set(h_update_button, 'Enable', 'On');
    end

    function cb_update(a, b)
        filt{1,1} = '*.mat'; filt{1,2} = 'Material file (*.mat)';
        [fname, pathname, filterindex] = uiputfile(filt, 'Save', material.name);
        if ~ischar(fname)
            %nothing selected
            fname = [];
            return;
        end
        save(fullfile(pathname, fname), 'material');
    end
end

function [ph_velocity, vel_poly, data2, theta, arrival_time, dist] = fn_calc_vel_poly(exp_data, thickness, z_min, z_max, max_fract_vel_shift, min_std_over_mean, poly_order, force_even_poly)
%first work out tx rx sep for each time trace
dx = abs(exp_data.array.el_xc(exp_data.tx) - exp_data.array.el_xc(exp_data.rx));
dxu = unique(dx);
dist = sqrt(dxu .^ 2 + (2 * thickness) ^ 2);
data2 = zeros(length(exp_data.time), length(dxu));
theta = atan(dxu / (2 * thickness));
vel = zeros(size(dxu));
arrival_time = zeros(size(dxu));
for ii = 1:length(dxu)
    jj = find(dx == dxu(ii));
    data2(:, ii) = mean(exp_data.time_data(:, jj), 2);
    if ii == 1
        z = exp_data.ph_velocity * exp_data.time / 2;
        [dummy, kk] = max(abs(data2(:, ii)) .* (z > z_min) .* (z < z_max));
        arrival_time(ii) = exp_data.time(kk);
        vel(ii) = dist(ii) / arrival_time(ii);
    else
        q = abs(data2(k1:k2, ii));
        %test mean / std
        if std(q) / mean(q) > min_std_over_mean
            [dummy, kk] = max(q);
            kk = kk + k1 - 1;
            arrival_time(ii) = exp_data.time(kk);
            vel(ii) = dist(ii) / arrival_time(ii);
        end
    end
    %search limits for next one
    k1 = round(kk * (1 - max_fract_vel_shift));
    k2 = round(kk * (1 + max_fract_vel_shift));
end


if force_even_poly
    theta2 = [-fliplr(theta), theta(2:end)];
    vel2 = [fliplr(vel), vel(2:end)];
else
    theta2 = theta;
    vel2 = vel;
end

ph_velocity = vel(1);
ii = find(vel2 > 0);
[vel_poly.p, dummy, vel_poly.mu] = polyfit(theta2(ii), vel2(ii), poly_order);
end
