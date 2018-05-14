function info = fn_2d_sizing_wrapper(exp_data, data, options)
if isempty(exp_data) & isempty(data)
    info.name = '2D direct image sizing';
    return;
else
    info = [];
end;

if ~isfield(options, 'select') || size(options.select, 1) ~= 2 || size(options.select, 2) ~= 2
    warndlg('Select image region first','Warning')
    return;
end

%default params
default_ang_pts = 91;
default_min_image_pts = 200;
default_db_down = 6;
default_method = 'Minimise area';

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
    'WindowStyle', 'Modal', ...
    'Name', ['Analysis:', '2D direct image sizing'] ...
);

[h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(f, table_pos, 'normalized', @fn_new_params);
content_info.db_down.label = 'Threshold (dB down)';
content_info.db_down.default = default_db_down;
content_info.db_down.type = 'double';
content_info.db_down.constraint = [1, 100];
content_info.db_down.multiplier = 1;

content_info.min_image_pts.label = 'Minimum image pixels';
content_info.min_image_pts.default = default_min_image_pts;
content_info.min_image_pts.type = 'int';
content_info.min_image_pts.constraint = [10, 1000];
content_info.min_image_pts.multiplier = 1;

content_info.ang_pts.label = 'Angular points';
content_info.ang_pts.default = default_ang_pts;
content_info.ang_pts.type = 'int';
content_info.ang_pts.constraint = [10, 901];
content_info.ang_pts.multiplier = 1;

content_info.method.label = 'Method';
content_info.method.default = default_method;
content_info.method.type = 'constrained';
content_info.method.constraint = {'Minimise area', 'Maximise length'};


h_fn_set_content(content_info);

h_result = uicontrol('Style', 'text', 'Units', 'Normalized', 'Position', result_pos);

%special colourmap (b/w below threshold, colour above)
g = 1 - gray;
j = jet;
cmap = [g(1:size(g,1) / 2, :);j(size(j,1) / 2 + 1:end, :)];
axes('OuterPosition', graph_pos)

%trigger the calc to run immediately
h_data_changed();

    function fn_new_params(params)
        fn_do_calc(data, options, params.db_down, params.min_image_pts, params.ang_pts, params.method, cmap, h_result);
    end
end

function fn_do_calc(data, options, db_down, min_image_pts, ang_pts, method, cmap, h_result)
xs = options.select(:,1);
zs = options.select(:,2);

%extract key bit of data
i1 = find(data.x >= min(xs) & data.x <= max(xs));
i2 = find(data.z >= min(zs) & data.z <= max(zs));
xr = data.x(i1);
zr = data.z(i2);
fr = abs(data.f(i2, i1)); %note reversal of indices in f!

%up sample if required
% min_im_dimension = min([length(i1), length(i2)]);
if length(i1) < min_image_pts
    fr = interpft(fr, min_image_pts, 2);
    xr = [0:min_image_pts-1] * (xr(2) - xr(1)) * length(i1) / min_image_pts + xr(1);
end
if length(i2) < min_image_pts
    fr = interpft(fr, min_image_pts, 1);
    zr = [0:min_image_pts-1] * (zr(2) - zr(1)) * length(i2) / min_image_pts + zr(1);
end
fr = abs(fr);

%get the >-6db image
max_val = max(max(abs(fr)));
[i2_m, i1_m] = find(abs(fr) == max_val);
thresh_val = max_val * 10 ^ (-db_down / 20);
b = abs(fr) >= thresh_val;
fr = (fr < thresh_val) .* fr / thresh_val * 0.5 + (fr >= thresh_val) .* ((fr - thresh_val) / (max_val - thresh_val) * 0.5 + 0.5);

%convert >-ndb pts to just a list of coordinates
[j2, j1] = find(b);
x = xr(j1);
z = zr(j2);
[t, r] = cart2pol(x, z);

%fit a rectangle at angle a
a = linspace(0, pi / 2, ang_pts);
% area = zeros(size(a));
min_area = inf;
max_len = 0;
imagesc(xr, zr, fr);
for ii = 1:length(a)
    [xrect, zrect, area, len] = fn_find_rect(t, r, a(ii));
    %draw rectangle
    cla;
    imagesc(xr * 1e3, zr * 1e3, fr);
    hold on;
    fn_plot_result(xr, zr, fr, cmap, xrect, zrect, 'k');
    if area < min_area
        min_area = area;
        xrect_min_area = xrect;
        zrect_min_area = zrect;
    end
    if len > max_len
        max_len = len;
        xrect_max_len = xrect;
        zrect_max_len = zrect;
    end
    pause(0.0001);
end

cla;
imagesc(xr * 1e3, zr * 1e3, fr);
hold on;
switch method
    case 'Minimise area'
        [len1, ang1, cen1] = fn_plot_result(xr, zr, fr, cmap, xrect_min_area, zrect_min_area, 'b');
        set(h_result, 'String', ...
            {' ', ...
            'MIN RECT AREA METHOD', ...
            sprintf('Length: %.1f mm', len1 * 1e3), ...
            sprintf('Angle: %.1f degrees', ang1 * 180 / pi), ...
            sprintf('Centre: (%.1f, %.1f) mm', cen1(1) * 1e3, cen1(2) * 1e3)}, ...
            'ForegroundColor', 'b', ...
            'HorizontalAlignment', 'Left');
    case 'Maximise length'
        [len2, ang2, cen2] = fn_plot_result(xr, zr, fr, cmap, xrect_max_len, zrect_max_len, 'r');
        set(h_result, 'String', ...
            {' ', ...
            'MAX LENGTH METHOD', ...
            sprintf('Length: %.1f mm', len2 * 1e3), ...
            sprintf('Angle: %.1f degrees', ang2 * 180 / pi), ...
            sprintf('Centre: (%.1f, %.1f) mm', cen2(1) * 1e3, cen2(2) * 1e3)}, ...
            'ForegroundColor', 'r', ...
            'HorizontalAlignment', 'Left');
end

end

function [xrect, zrect, area, len] = fn_find_rect(t, r, a)
%rotate coordinates
[x_rot, z_rot] = pol2cart(t - a, r);
%find limits
xmin = min(x_rot);
xmax = max(x_rot);
zmin = min(z_rot);
zmax = max(z_rot);
area = (xmax - xmin) * (zmax - zmin);
len = max([xmax - xmin, zmax - zmin])
[trect, rrect] = cart2pol([xmin, xmax, xmax, xmin, xmin],[zmin, zmin, zmax, zmax, zmin]);
[xrect, zrect] = pol2cart(trect + a, rrect);
end

function [len, ang, cen] = fn_plot_result(xr, zr, fr, cmap, xrect, zrect, col)
% cla;
% imagesc(xr * 1e3, zr * 1e3, fr);
% hold on;
plot(xrect * 1e3, zrect * 1e3, [col,':']);
axis equal; axis tight;
colormap(cmap);
d1 = sqrt((xrect(2) - xrect(1)) ^ 2 + (zrect(2) - zrect(1)) ^ 2);
d2 = sqrt((xrect(3) - xrect(2)) ^ 2 + (zrect(3) - zrect(2)) ^ 2);
cx = mean(xrect(1:4));
cz = mean(zrect(1:4));
a = atan2(zrect(2) - zrect(1), xrect(2) - xrect(1)); %angle of first side w.r.t. x axis
im = min([(max(xr) - min(xr)), (max(zr) - min(zr))]) / 4;
plot(cx * [1,1] * 1e3, [cz, cz - im] * 1e3, col);
if d1 > d2
    len = d1;
    ang = a;
    px = xrect(1:2) - 0.5 * (xrect(2) - xrect(3));
    pz = zrect(1:2) - 0.5 * (zrect(2) - zrect(3));
else
    len = d2;
    ang = a - pi / 2;
    px = xrect(2:3) - 0.5 * (xrect(2) - xrect(1));
    pz = zrect(2:3) - 0.5 * (zrect(2) - zrect(1));
end;
aa = linspace(0, ang, 100);
plot(px * 1e3, pz * 1e3, [col, '.-']);
plot(cx * 1e3, cz * 1e3, [col, 'o']);
plot([cx, cx + im * cos(ang - pi / 2)] * 1e3, [cz, cz + im * sin(ang - pi / 2)]  * 1e3, col);
plot([cx, cx + 0.75 * im * cos(aa - pi / 2)] * 1e3, [cz, cz + 0.75 * im * sin(aa - pi / 2)] * 1e3, col);
axis([min(xr), max(xr), min(zr), max(zr)] * 1e3);
cen = [cx, cz];
xlabel('X (mm)');
ylabel('Z (mm)');
end
