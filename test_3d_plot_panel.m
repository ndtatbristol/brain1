fn_clear;
load('Instruments\Emulator data\3MHz 128el 2D array aluminium HMC.mat');
addpath('.\Imaging');
addpath('.\gpu stuff');

options.x_size = 30e-3;
options.y_size = 30e-3;
options.z_size = 50e-3;
options.x_offset = 0;
options.y_offset = 0;
options.z_offset = 5e03;
options.pixel_size = 1e-3;
options.angle_limit_on = 0;
options.interpolation_method = 'nearest';
options.centre_freq = 3e6;
options.frac_half_bandwidth = 1;
options.filter_on = 1;

data = fn_basic_plusgpu_wrapper(exp_data, options, 'recalc_and_process');

f = figure('MenuBar', 'none', 'ToolBar', 'None', 'Name', 'Test NDT 3D display panel');
panel_rel_pos = [0, 0, 1, 1];
panel_colour = get(gcf, 'Color');
h_panel = uipanel('BorderType','none',...
    'BackgroundColor', panel_colour,...
    'Units', 'normalized',...
    'Position', panel_rel_pos,...
    'Parent', f);
h_toolbar = uitoolbar(f);

%turn the panel into a plot panel
[h_fn_update_data, h_fn_get_options, h_fn_set_options] = gui_3d_plot_panel(h_panel, h_toolbar);

%add the data
% h_fn_update_data(data);

tmp = data.f;
% while 1
%     data.f = tmp + (randn(size(data.f)) + i * randn(size(data.f))) * 0.1;
    h_fn_update_data(data);
% end



% %test some 3D visualisation ideas
% fn_clear;
% x = linspace(-1,1,50);
% [X,Y,Z] = meshgrid(x,x,x);
% 
% R = sqrt(X .^ 2 - Y .^ 2 + Z .^ 2);
% 
% fv = isosurface(X, Y, Z, R, 0.5, 'NoShare');
% % fv2 = isocaps(X, Y, Z, R, 0.5, 'NoShare', 'below');
% 
% figure;
% subplot(2,2,1);
% patch(fv, 'FaceColor' ,'r', 'EdgeColor','none');
% % hold on;
% % patch(fv2, 'FaceColor','interp', 'EdgeColor','none');
% view(3);
% axis equal;
% camlight 
% lighting gouraud
% alpha(0.1);
% 
