fn_clear;
load('test_2d_image.mat');

%create figure with panel in it
f = figure('MenuBar', 'none', 'ToolBar', 'None', 'Name', 'Test NDT 2D display panel');
left_panel_rel_pos = [0, 0, 1, 1];
panel_colour = get(gcf, 'Color');
h_panel = uipanel('BorderType','none',...
    'BackgroundColor', panel_colour,...
    'Units', 'normalized',...
    'Position', left_panel_rel_pos,...
    'Parent', f);
h_toolbar= uitoolbar(f);

%turn the panel into a plot panel
[h_fn_update_data, h_fn_get_options, h_fn_set_options] = gui_2d_plot_panel(h_panel, h_toolbar);

%stick in the data
tmp = data.f;
while 1
    data.f = tmp + (randn(size(data.f)) + i * randn(size(data.f))) * 0.1;
    h_fn_update_data(data);
end
