function info = fn_surface_breaking_characterisation(exp_data, data, options)
if isempty(exp_data) & isempty(data)
    info.name = 'Surface breaking crack characterisation';
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
default_centre_freq = exp_data.array.centre_freq;
default_el_width = 2.*(exp_data.array.el_x1(1)-exp_data.array.el_x2(1));
default_correct_for_propagation_dist = 1;
default_correct_for_el_directivity = 1;
default_disp_manifold = 0;

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

content_info.disp_manifold.label = 'Display defect manifold';
content_info.disp_manifold.default = default_disp_manifold;
content_info.disp_manifold.type = 'bool';
content_info.disp_manifold.constraint = {'On', 'Off'};

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
        options.display_manifold = params.disp_manifold;
        
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
        
        [prob_map_plot1, T_par_crack, p_par_crack, par_test1]=fn_defect_char(s, exp_data, options);
        TR=triangulation(T_par_crack,[p_par_crack prob_map_plot1]);
        ic=incenter(TR);
        F = scatteredInterpolant(ic(:,1),ic(:,2),ic(:,3));
        int_x=linspace(min(p_par_crack(:,1)),max(p_par_crack(:,1)),100);
        int_y=linspace(min(p_par_crack(:,2)),max(p_par_crack(:,2)),100);
        [mesh_x mesh_y]=meshgrid(int_x, int_y);
        int_c = F(mesh_x,mesh_y);
        axes(ax_gr);
        cla;
        imagesc(int_x,int_y,int_c)
        
        %patch('Faces',T_par_crack,'Vertices',p_par_crack,'FaceColor','interp','FaceVertexCData',(prob_map_plot1)/max(abs(prob_map_plot1(:))),'LineStyle','none');
        colorbar;
        %axis square
        axis tight
        ylabel('orientation angle (\circ)')
        xlabel('size (\lambda)')
        hold on
        plot(par_test1(1), par_test1(2), 'ro', 'LineWidth', 2)
        
        str = 'Most likely scatterer:';
        str2 = {...
                sprintf('Angle: %i degrees', round(par_test1(2))), ...
                sprintf('Length: %.2f wavelengths', par_test1(1))};
            
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

