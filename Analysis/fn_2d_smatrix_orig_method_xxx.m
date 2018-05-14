function info = fn_2d_smatrix_orig_method_wrapper(exp_data, data, options)
if isempty(exp_data) & isempty(data)
    info.name = '2D S-matrix (original method)';
    return;
else
    info = [];
end;

if ~isfield(options, 'select') || size(options.select, 1) ~= 2 || size(options.select, 2) ~= 2
    warndlg('Select image region first','Warning')
    return;
end

%find max in selected part of image passed to routine - this is the point where S-matrix is
%calculated
i1 = min(find(data.x >= min(options.select(:,1))));
i2 = max(find(data.x <= max(options.select(:,1))));
j1 = min(find(data.z >= min(options.select(:,2))));
j2 = max(find(data.z <= max(options.select(:,2))));
data.x = data.x(i1:i2);
data.z = data.z(j1:j2);
data.f = data.f(j1:j2, i1:i2);
% keyboard;
[dummy, ii] = max(abs(data.f));
[dummy, jj] = max(dummy);
ii = ii(jj);
options.x = data.x(jj);
options.z = data.z(ii);

%defaults
default_aperture_elements = 8;
default_ang_pts = 90;
default_density = 2700;
default_speed_ratio = 2;
default_fract_bandwidth = 1;
default_centre_freq = 5e6;
default_el_width = 0.53e-3;
default_correct_for_propagation_dist = 1;
default_correct_for_el_directivity = 1;

% pad_factor = default_pad_factor;
% ang_pts = default_ang_pts;

%figure size
width = 600;
height = 300;
table_pos = [0,1/2,1/3,1/2];
result_pos = [0,0,1/3,1/2];
smatrix_pos = [1/3,0,1/3,1];
graph_pos = [2/3,0,1/3,1];

%create figure
p = get(0, 'ScreenSize');
f = figure('Position',[(p(3) - width) / 2, (p(4) - height) / 2, width, height] ,...
    'MenuBar', 'none', ...
    'NumberTitle', 'off', ...
    'ToolBar', 'None', ...
    'Name', ['Analysis:', '2D S-matrix extraction (original method)'] ...
    );

%create graph panels
h_smatrix_panel = uipanel(f, 'Units', 'Normalized', 'Position', smatrix_pos);
h_graph_panel = uipanel(f, 'Units', 'Normalized', 'Position', graph_pos);

%results
h_result = uicontrol('Style', 'text', 'Units', 'Normalized', 'Position', result_pos);

[h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(f, table_pos, 'normalized', @fn_new_params);

content_info.centre_freq.label = 'Centre frequency (MHz)';
content_info.centre_freq.default = default_centre_freq;
content_info.centre_freq.type = 'double';
content_info.centre_freq.constraint = [1, 1e12];
content_info.centre_freq.multiplier = 1e6;

content_info.aperture_els.label = 'Aperture elements';
content_info.aperture_els.default = default_aperture_elements;
content_info.aperture_els.type = 'int';
content_info.aperture_els.constraint = [1, length(exp_data.array.el_xc) - 1];

content_info.fract_bandwidth.label = 'Bandwidth (%)';
content_info.fract_bandwidth.default = default_fract_bandwidth;
content_info.fract_bandwidth.type = 'double';
content_info.fract_bandwidth.constraint = [0.01, 10];
content_info.fract_bandwidth.multiplier = 0.01;

content_info.el_width.label = 'Element width (mm)';
if isfield(exp_data.array, 'el_x2') %this is not always defined
    content_info.el_width.default = abs(max([exp_data.array.el_x2(1) - exp_data.array.el_xc(1), ...
        exp_data.array.el_x1(1) - exp_data.array.el_xc(1)])) * 2;
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

content_info.correct_for_el_directivity.label = 'Correct for element directivity';
content_info.correct_for_el_directivity.default = default_correct_for_el_directivity;
content_info.correct_for_el_directivity.type = 'bool';
content_info.correct_for_el_directivity.constraint = {'On', 'Off'};

content_info.correct_for_propagation_dist.label = 'Correct for propagation distance';
content_info.correct_for_propagation_dist.default = default_correct_for_propagation_dist;
content_info.correct_for_propagation_dist.type = 'bool';
content_info.correct_for_propagation_dist.constraint = {'On', 'Off'};

h_fn_set_content(content_info);

h_result = uicontrol('Style', 'text', 'Units', 'Normalized', 'Position', result_pos);

ax_sm = axes('Parent', h_smatrix_panel);
ax_gr = axes('Parent', h_graph_panel);

%trigger the calc
h_data_changed();

    function fn_new_params(params)
        options.fract_bandwidth = params.fract_bandwidth;
        options.centre_freq = params.centre_freq;
        options.aperture_els = params.aperture_els;
        options.speed_ratio = params.speed_ratio;
        options.correct_for_propagation_dist = params.correct_for_propagation_dist;
        options.correct_for_el_directivity = params.correct_for_el_directivity;
        options.el_width = params.el_width;
        s = fn_2d_s_matrix_orig_method(exp_data, options);
        
        axes(ax_sm);
        cla;
        pcolor(s.phi * 180 / pi, s.phi * 180 / pi, abs(s.m));
        shading flat;
%         c = caxis;
%         caxis([0, c(2)]);
        axis equal;
        axis tight;
        axis xy;
        xlabel('{\theta}_1');
        ylabel('{\theta}_2');
        hold on;
        
        axes(ax_gr);
        cla;
        di = abs(diag(s.m));
        plot(s.phi * 180 / pi, di);
        axis tight;
        hold on;
        plot([min(s.phi), max(s.phi)] * 180 / pi, [1, 1] * max(di) / 2, 'r');
        [max_di, i2] = max(di);
        i1 = min(find(di > max_di / 2));
        i3 = max(find(di > max_di / 2));
        plot(s.phi(i1) * [1, 1] * 180 / pi, [0, max(di) / 2], 'r');
        plot(s.phi(i2) * [1, 1] * 180 / pi, [0, max(di)], 'r');
        plot(s.phi(i3) * [1, 1] * 180 / pi, [0, max(di) / 2], 'r');
        failed = 0;
        if (i2 == 1) | (i2 == length(di))
            str = 'FAILED: peak out of range';
            str2 = ' ';
        else
            if i1 > 1 & i3 < length(di)
                str = 'Both HM in range';
                dphi = abs(s.phi(i3) - s.phi(i1)) / 2;
            else
                if i1 == 1;
                    str = 'Lower HM point out of range';
                    dphi = abs(s.phi(i3) - s.phi(i2));
                end
                if i3 == length(di)
                    str = 'Upper HM point out of range';
                    dphi = abs(s.phi(i2) - s.phi(i1));
                end
            end
            wavelength = exp_data.ph_velocity / params.centre_freq;
            crack_length = fn_crack_length_from_hwhm(dphi) * wavelength;
            str2 = {...
                sprintf('Angle: %i degrees', round(s.phi(i2) * 180 / pi)), ...
                sprintf('HWHM: %i degrees', round(dphi * 180 / pi)), ...
                sprintf('Length: %.2f mm', crack_length * 1e3)};
            axes(ax_sm);
            plot([min(s.phi), max(s.phi)] * 180 / pi, [min(s.phi), max(s.phi)] * 180 / pi, 'w:');
            plot(s.phi(i2) * 180 / pi, s.phi(i2) * 180 / pi, 'wo');
        end
        
        set(h_result, 'String', ...
            {' ', str, str2{:}}, ...
            'ForegroundColor', 'r', ...
            'HorizontalAlignment', 'Left');
    end
end

function s = fn_2d_s_matrix_orig_method(exp_data, options)
mesh.x = options.x;
mesh.z = options.z;
TFM_focal_law = fn_calc_tfm_focal_law2(exp_data, mesh);
TFM_focal_law.interpolation_method = 'linear';
TFM_focal_law.filter_on = 1;
TFM_focal_law.filter = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.fract_bandwidth / 2);
TFM_focal_law.lookup_time_tx = TFM_focal_law.lookup_time;
TFM_focal_law.lookup_time_rx = TFM_focal_law.lookup_time;
TFM_focal_law.lookup_ind_tx = TFM_focal_law.lookup_ind;
TFM_focal_law.lookup_ind_rx = TFM_focal_law.lookup_ind;

lookup_amp = ones(size(TFM_focal_law.lookup_amp)); %store this for later (should be corrected for directivity at this point
TFM_focal_law = rmfield(TFM_focal_law, {'lookup_amp', 'lookup_ind', 'lookup_time'});
%work out apertures
i1 = 1:length(exp_data.array.el_xc) - options.aperture_els;
i2 = i1 + options.aperture_els;
x1 = exp_data.array.el_xc(i1);
x2 = exp_data.array.el_xc(i2);
s.phi = atan2(options.x - (x2 + x1) / 2, options.z);

s.m = zeros(length(i1), length(i2));
%filter now to save doing every time
n = length(TFM_focal_law.filter);
exp_data.time_data = ifft(spdiags(TFM_focal_law.filter, 0, n, n) * fft(exp_data.time_data));
TFM_focal_law.filter_on = 0;
TFM_focal_law.hilbert_on = 0;

d1 = sqrt((exp_data.array.el_xc - options.x) .^ 2 + (exp_data.array.el_zc - options.z) .^ 2);
q = zeros(length(exp_data.array.el_xc), length(exp_data.array.el_xc));
d2 = d1(exp_data.tx) + d1(exp_data.rx);

vals = zeros(size(d2));
for ii = 1:length(vals)
    vals(ii) = interp1(exp_data.time * exp_data.ph_velocity, exp_data.time_data(:, ii), d2(ii), 'linear', 0);
end

if options.correct_for_propagation_dist
    vals = vals .* sqrt(d1(exp_data.tx)) .* sqrt(d1(exp_data.rx));
end

if options.correct_for_el_directivity
    %approximation to half-space directivity function
    theta = atan2(exp_data.array.el_xc - options.x, exp_data.array.el_zc - options.z);
    vals = vals ./ cos(theta(exp_data.tx)) ./ cos(theta(exp_data.rx));
    %and the element width effect
    a = options.el_width;
    lambda = exp_data.ph_velocity / options.centre_freq;
    vals = vals ./ sinc(a * sin(theta(exp_data.tx)) / lambda) ./ sinc(a * sin(theta(exp_data.rx)) / lambda);
end

%put into a matrix
q = zeros(length(exp_data.array.el_xc), length(exp_data.array.el_xc));
for ii = 1:length(exp_data.tx)
    q(exp_data.tx(ii), exp_data.rx(ii)) = vals(ii);
end

%deal with HMC data if nesc
if ~any(any(tril(q,-1)))
    q = q + triu(q, 1).';
end;

a = [ones(options.aperture_els, size(q,1));zeros(size(q,1)-options.aperture_els+1,size(q,1))];
a = reshape(a, size(q,1), []);
a = a(:, 1:size(q,1) - options.aperture_els);

s.m = a' * q * a;
end

function crack_length = fn_crack_length_from_hwhm(hwhm)
data = [
    0.0500    0.9460
    0.1000    0.9193
    0.1500    0.9014
    0.2000    0.8836
    0.2500    0.8657
    0.3000    0.8300
    0.3500    0.7586
    0.4000    0.6515
    0.4500    0.5801
    0.5000    0.5444
    0.5500    0.5176
    0.6000    0.4819
    0.6500    0.4462
    0.7000    0.4284
    0.7500    0.4016
    0.8000    0.3748
    0.8500    0.3391
    0.9000    0.3124
    0.9500    0.3034
    1.0000    0.2856
    1.0500    0.2767
    1.1000    0.2677
    1.1500    0.2588
    1.2000    0.2588
    1.2500    0.2410
    1.3000    0.2320
    1.3500    0.2142
    1.4000    0.2053
    1.4500    0.1963
    1.5000    0.1963
    1.5500    0.1874
    1.6000    0.1874
    1.6500    0.1785
    1.7000    0.1785
    1.7500    0.1696
    1.8000    0.1606
    1.8500    0.1517
    1.9000    0.1517
    1.9500    0.1517
    2.0000    0.1428
    2.0500    0.1428
    2.1000    0.1428
    2.1500    0.1428
    2.2000    0.1339
    2.2500    0.1339
    2.3000    0.1249
    2.3500    0.1249
    2.4000    0.1160
    2.4500    0.1160
    2.5000    0.1160
    2.5500    0.1160
    2.6000    0.1160
    2.6500    0.1071
    2.7000    0.1071
    2.7500    0.1071
    2.8000    0.0982
    2.8500    0.0982
    2.9000    0.0982
    2.9500    0.0982
    3.0000    0.0982
    3.0500    0.0982
    3.1000    0.0982
    3.1500    0.0892
    3.2000    0.0892
    3.2500    0.0892
    3.3000    0.0892
    3.3500    0.0892
    3.4000    0.0803
    3.4500    0.0803
    3.5000    0.0803
    3.5500    0.0803
    3.6000    0.0803
    3.6500    0.0803
    3.7000    0.0803
    3.7500    0.0803
    3.8000    0.0803
    3.8500    0.0714
    3.9000    0.0714
    3.9500    0.0714
    4.0000    0.0714
    4.0500    0.0714
    4.1000    0.0714
    4.1500    0.0714
    4.2000    0.0714
    4.2500    0.0714
    4.3000    0.0714
    4.3500    0.0714
    4.4000    0.0714
    4.4500    0.0714
    4.5000    0.0714
    4.5500    0.0714
    4.6000    0.0714
    4.6500    0.0714
    4.7000    0.0714
    4.7500    0.0714
    4.8000    0.0714
    4.8500    0.0714
    4.9000    0.0714
    4.9500    0.0714
    5.0000    0.0714];

if hwhm < min(data(:,2))
    crack_length = min(data(:,1));
    return;
end
if hwhm > max(data(:,2))
    crack_length = max(data(:,1));
    return;
end
crack_length = data(max(find(data(:,2) > hwhm)), 1);
end