function [material, fname] = fn_input_new_material_details(materials_path)
%opens window to enable user to input material properties for a material
%that will be added to list and file added in materials_path directory

config = fn_get_config;

default_velocity = 1000;
default_name = 'New material';

material.name = default_name;
material.vel_spherical_harmonic_coeffs = default_velocity;
fname = fn_get_fname;

%create figure
f = figure('Position', ...
        [(config.general.screen_size_pixels(3) - config.new_matl_win.pixel_size(1)) / 2, ...
        (config.general.screen_size_pixels(4) - config.new_matl_win.pixel_size(2)) / 2, ...
        config.new_matl_win.pixel_size(1), ...
        config.new_matl_win.pixel_size(2)] ,...
    'MenuBar', 'none', ...
    'NumberTitle', 'off', ...
    'ToolBar', 'None', ...
    'WindowStyle', 'Modal', ...
    'Name', 'Create new material' ...
);

h_ok_btn = uicontrol('Style', 'PushButton', 'String', 'OK', 'Callback', @fn_ok);
h_cancel_btn = uicontrol('Style', 'PushButton', 'String', 'Cancel', 'Callback', @fn_cancel);

[h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(f, [0,0,1,1], 'normalized', @fn_new_params);
content_info.name.label = 'Name';
content_info.name.default = default_name;
content_info.name.type = 'string';

content_info.velocity.label = 'Velocity (m/s)';
content_info.velocity.default = default_velocity;
content_info.velocity.type = 'double';
content_info.velocity.constraint = [0.01, 100000];
content_info.velocity.multiplier = 1;

h_fn_set_content(content_info);
set(f, 'Visible', 'On', 'ResizeFcn', @fn_window_resize);
fn_window_resize([], []);
uiwait(f);

    function fn_new_params(params)
        material.name = params.name;
        material.vel_spherical_harmonic_coeffs = params.velocity;
        fname = fn_get_fname;
    end

    function fn_save
        fname = fn_get_fname;
        save([materials_path, filesep, fname, '.mat'], 'material');
    end

    function fname = fn_get_fname
        [v_mean, v_x, v_y, v_z]= fn_get_nominal_velocity(material.vel_spherical_harmonic_coeffs);
        if (v_x ~= v_mean) || (v_y ~= v_mean) || (v_z ~= v_mean)
            s = ' variable';
        else
            s ='';
        end
        fname = sprintf([material.name, ' (%i', s, ')'], round(v_mean));
    end

    function fn_window_resize(x, y)
        p = getpixelposition(f);
        setpixelposition(h_ok_btn, [1, 1, p(3) / 2, config.general.button_height_pixels]);
        setpixelposition(h_cancel_btn, [p(3)/2, 1, p(3) / 2, config.general.button_height_pixels]);
        setpixelposition(h_table, [1, config.general.button_height_pixels, p(3), p(4) - config.general.button_height_pixels]);
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