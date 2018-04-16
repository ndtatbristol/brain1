function [array, fname] = fn_input_new_array_details(arrays_path)
%opens window to enable user to input array
%that will be added to list and file added in arrays_path directory

config = fn_get_config;

%create figure
f = figure('Position', ...
        [(config.general.screen_size_pixels(3) - config.new_array_win.pixel_size(1)) / 2, ...
        (config.general.screen_size_pixels(4) - config.new_array_win.pixel_size(2)) / 2, ...
        config.new_array_win.pixel_size(1), ...
        config.new_array_win.pixel_size(2)] ,...
    'MenuBar', 'none', ...
    'NumberTitle', 'off', ...
    'ToolBar', 'None', ...
    'WindowStyle', 'Modal', ...
    'Name', 'Create new array', ...
    'Visible', 'Off', ...
    'ResizeFcn', @fn_window_resize ...
);

h_ok_btn = uicontrol('Style', 'PushButton', 'String', 'OK', 'Callback', @fn_ok);
h_cancel_btn = uicontrol('Style', 'PushButton', 'String', 'Cancel', 'Callback', @fn_cancel);

[h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(f, [0,0,1,1], 'normalized', @fn_new_params);

h_axes = axes;

content_info.no_els.label = 'Number of elements';
content_info.no_els.default = config.new_array_win.default_no_els;
content_info.no_els.type = 'int';
content_info.no_els.constraint = [1, 10000];

content_info.manufacturer.label = 'Manufacturer';
content_info.manufacturer.default = config.new_array_win.default_manufacturer;
content_info.manufacturer.type = 'string';

content_info.centre_freq.label = 'Nominal cent. freq. (MHz)';
content_info.centre_freq.default = config.new_array_win.default_cent_freq;
content_info.centre_freq.type = 'double';
content_info.centre_freq.constraint = [1e3, 1e9];
content_info.centre_freq.multiplier = 1e6;

content_info.pitch.label = 'Element pitch (mm)';
content_info.pitch.default = config.new_array_win.default_pitch;
content_info.pitch.type = 'double';
content_info.pitch.constraint = [1e-6, 1];
content_info.pitch.multiplier = 1e-3;

content_info.separation.label = 'Element separation (mm)';
content_info.separation.default = config.new_array_win.default_separation;
content_info.separation.type = 'double';
content_info.separation.constraint = [1e-6, 1];
content_info.separation.multiplier = 1e-3;

content_info.length.label = 'Element length (mm)';
content_info.length.default = config.new_array_win.default_length;
content_info.length.type = 'double';
content_info.length.constraint = [1e-6, 1];
content_info.length.multiplier = 1e-3;

h_fn_set_content(content_info);

params = h_fn_get_data();
fn_new_params(params);

set(f, 'Visible', 'On');


uiwait(f);

    function fn_new_params(params)
        array.manufacturer = params.manufacturer;
        array.centre_freq = params.centre_freq;
        width = params.pitch - params.separation;
        array.el_xc = [1:params.no_els] * params.pitch;
        array.el_xc = array.el_xc - mean(array.el_xc);
        array.el_yc = zeros(size(array.el_xc));
        array.el_zc = zeros(size(array.el_xc));
        array.el_x1 = array.el_xc + width / 2;
        array.el_y1 = zeros(size(array.el_xc));
        array.el_z1 = zeros(size(array.el_xc));
        array.el_x2 = array.el_xc;
        array.el_y2 = array.el_yc + params.length / 2;
        array.el_z2 = zeros(size(array.el_xc));
        
        fname = strtrim(fn_generate_array_filename(array));
        cla;
        array_geom = fn_get_array_geom_for_plots(array);
        patch(array_geom.x, array_geom.y, config.array_el_patch_color, 'EdgeColor', config.array_el_edge_color);
        axis equal;
        axis off;
    end

    function fn_save
        fname = fn_generate_array_filename(array);
        save([arrays_path, filesep, fname, '.mat'], 'array');
    end

    function fn_window_resize(x, y)
        p = getpixelposition(f);
        tw = p(3) * config.new_array_win.table_frac;
        setpixelposition(h_ok_btn, [1, 1, tw / 2, config.general.button_height_pixels]);
        setpixelposition(h_cancel_btn, [tw / 2, 1, tw / 2, config.general.button_height_pixels]);
        setpixelposition(h_table, [1, config.general.button_height_pixels, tw, p(4) - config.general.button_height_pixels]);
        set(h_axes, 'Units', 'Pixels', 'OuterPosition', [tw + 1, 1, p(3) - tw, p(4)]);
    end

    function fn_ok(x, y)
        fn_save;
        close(f);
    end

    function fn_cancel(x, y)
        material = [];
        close(f);
    end
end