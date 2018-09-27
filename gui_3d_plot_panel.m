function [h_fn_update_data, h_fn_get_options, h_fn_set_options] = gui_2d_plot_panel(h_panel, h_toolbar)
default_options.max_val = [];
default_options.db_range = [];
default_options.global_max = [];
default_options.norm_val = [];
default_options.x_lim = [];
default_options.y_lim = [];
default_options.z_lim = [];
default_options.select = [];
default_options.x_plane = [];
default_options.y_plane = [];
default_options.z_plane = [];
default_options.active_plane = [];
default_options.show_sections = 0;

default_options.x_axis_sf = 1000;
default_options.y_axis_sf = 1000;
default_options.z_axis_sf = 1000;
default_options.cursor_type = 'none';
default_options.auto_normalise = 1;

options = [];
options = fn_set_default_fields(options, default_options);

% data_has_been_plotted = 0;

%load configuration data
config = fn_get_config;

next_point = 1;
pointer_str = '';
range_str = '';

plot_data = [];

% cursor_type = 'cross hair';

%load icons
icons = [];
load(config.files.icon_file);

%define all the objects
h_panels = fn_create_panels;

%add the axes to the graph panels and link appropriate ones
h_axes = fn_create_axes;

%add the actual plot objects (these always exist, they are just updated in
%position/visibility in operation)
h_plot_objects = fn_create_plot_objects;

%various handles
h_toolbar_btns = [];
h_max_slider = [];
h_range_slider = [];

%create the controls
fn_create_controls;

%handle for function that gets and sets options
h_fn_get_options = @fn_get_options;
h_fn_set_options = @fn_set_options;

%work through the options and click appropriate buttons etc.
fn_set_controls_to_match_options;

%force an update of the layout (which sets the relative positions of the
%different panels) and a resize
fn_resize;

%populate the returned function handles
h_fn_update_data = @fn_update_data;

%redefine the resize function of the container panel
set(h_panel, 'ResizeFcn', @fn_resize);

h_zoom = zoom;
h_pan = pan;

%--------------------------------------------------------------------------

    function ops = fn_get_options(dummy) %this function returns all current options settings and is designed to be called externally
        fn_set_options_to_match_toolbar_btns; %this should force update of all options fields according to view, controls etc
        ops = options;
    end

    function fn_set_options_to_match_toolbar_btns %read the status of all controls and set corresponding option fields
        %set scale
        if fn_get_control_status(h_toolbar, 'scale.linear')
            options.scale_mode = 'linear';
        end;
        if fn_get_control_status(h_toolbar, 'scale.log')
            options.scale_mode = 'log';
        end;
        %set plot value
        if fn_get_control_status(h_toolbar, 'plotwhat.real')
            options.plotwhat = 'real';
        end;
        if fn_get_control_status(h_toolbar, 'plotwhat.mod')
            options.plotwhat = 'mod';
        end;
        if fn_get_control_status(h_toolbar, 'plotwhat.arg')
            options.plotwhat = 'arg';
        end;
        %set cursor type
        if fn_get_control_status(h_toolbar, 'cursor.point')
            options.cursor_type = 'point';
        elseif fn_get_control_status(h_toolbar, 'cursor.region')
            options.cursor_type = 'region';
        else
            options.cursor_type = 'none';
        end
    end

    function fn_set_options(new_options)
        options = fn_set_default_fields(new_options, default_options);
        %         fn_set_custom_buttons;
        fn_set_controls_to_match_options;
        fn_update_graphs;
    end

    function fn_set_controls_to_match_options %sets controls to match options - called when panel is created
        switch options.cursor_type
            case 'point'
                fn_set_control_status(h_toolbar, 'cursor.point', 1);
                fn_set_control_status(h_toolbar, 'cursor.region', 0);
            case 'region'
                fn_set_control_status(h_toolbar, 'cursor.point', 0);
                fn_set_control_status(h_toolbar, 'cursor.region', 1);
        end
        if options.auto_normalise
            fn_set_control_status(h_toolbar, 'auto_normalise', 1);
        else
            fn_set_control_status(h_toolbar, 'auto_normalise', 0);
        end
        %         if options.show_colorbar
        %             fn_set_control_status(h_toolbar, 'colorbar', 1);
        %         else
        %             fn_set_control_status(h_toolbar, 'colorbar', 0);
        %         end
        %         fn_set_sliders;
    end

    function fn_create_controls
        %buttons in toolbar
        ii = 1;
        %         h_toolbar_btns(ii).separator = 1;
        %         h_toolbar_btns(ii).tag = 'view.zoomin';
        %         h_toolbar_btns(ii).tooltip = 'Zoom in';
        %         h_toolbar_btns(ii).icon = fn_get_cdata_for_named_icon(icons, 'Exploration.ZoomIn');
        %         h_toolbar_btns(ii).type = 'toggle';
        %         ii = ii+ 1;
        %
        %         h_toolbar_btns(ii).tag = 'zoomout';
        %         h_toolbar_btns(ii).tooltip = 'Zoom out';
        %         h_toolbar_btns(ii).icon = fn_get_cdata_for_named_icon(icons, 'Exploration.ZoomOut');
        %         h_toolbar_btns(ii).type = 'pushtool';
        %         ii = ii + 1;
        
        h_toolbar_btns(ii).tag = 'view.rotate';
        h_toolbar_btns(ii).tooltip = 'Rotate';
        h_toolbar_btns(ii).icon = fn_get_cdata_for_named_icon(icons, 'Exploration.Rotate');
        h_toolbar_btns(ii).type = 'toggle';
        ii = ii + 1;
        
        h_toolbar_btns(ii).separator = 1;
        h_toolbar_btns(ii).tag = 'cursor.point';
        h_toolbar_btns(ii).tooltip = 'Select point';
        h_toolbar_btns(ii).icon = fn_get_cdata_for_named_icon(icons, 'point');
        h_toolbar_btns(ii).type = 'toggle';
        ii = ii + 1;
        
        h_toolbar_btns(ii).tag = 'cursor.region';
        h_toolbar_btns(ii).tooltip = 'Select region';
        h_toolbar_btns(ii).icon = fn_get_cdata_for_named_icon(icons, 'region');
        h_toolbar_btns(ii).type = 'toggle';
        ii = ii + 1;
        
        h_toolbar_btns(ii).tag = 'normalise';
        h_toolbar_btns(ii).tooltip = 'Normalise';
        h_toolbar_btns(ii).icon = fn_get_cdata_for_named_icon(icons, 'normalise');
        h_toolbar_btns(ii).type = 'pushtool';
        ii = ii + 1;

        h_toolbar_btns(ii).tag = 'auto_normalise';
        h_toolbar_btns(ii).tooltip = 'Auto normalise';
        h_toolbar_btns(ii).icon = fn_get_cdata_for_named_icon(icons, 'auto_normalise');
        h_toolbar_btns(ii).type = 'toggle';
        ii = ii + 1;

        h_toolbar_btns(ii).tag = 'sections';
        h_toolbar_btns(ii).tooltip = 'Show sections';
        h_toolbar_btns(ii).icon = fn_get_cdata_for_named_icon(icons, 'show_sections');
        h_toolbar_btns(ii).type = 'toggle';
        ii = ii + 1;
        
        %         keyboard
        %create toolbar controls using above
        for ii = 1:length(h_toolbar_btns)
            if isfield(h_toolbar_btns(ii), 'separator') & h_toolbar_btns(ii).separator
                sep = 'On';
            else
                sep = 'Off';
            end;
            switch h_toolbar_btns(ii).type
                case 'toggle'
                    h_toolbar_btns(ii).handle = uitoggletool(h_toolbar, ...
                        'CData', h_toolbar_btns(ii).icon, ...
                        'TooltipString', h_toolbar_btns(ii).tooltip, ...
                        'Tag', h_toolbar_btns(ii).tag, ...
                        'HandleVisibility', 'Off', ...
                        'ClickedCallback', @cb_control, ...
                        'Separator', sep);
                case 'pushtool'
                    h_toolbar_btns(ii).handle = uipushtool(h_toolbar, ...
                        'CData', h_toolbar_btns(ii).icon, ...
                        'TooltipString', h_toolbar_btns(ii).tooltip, ...
                        'Tag', h_toolbar_btns(ii).tag, ...
                        'HandleVisibility', 'Off', ...
                        'ClickedCallback', @cb_control, ...
                        'Separator', sep);
            end;
        end;
        %add range sliders
        h_max_slider = uicontrol(h_panels.range, ...
            'Style', 'slider', 'String', '37', ...
            'Callback', @cb_control, 'Min', 0, 'Max', 100, 'Value', 40, ...
            'Tag', 'max_value');
        h_range_slider = uicontrol(h_panels.range, ...
            'Style', 'slider', 'String', '37', ...
            'Callback', @cb_control, 'Min', 0, 'Max', 100, 'Value', 40, ...
            'Tag', 'range');
    end

    function fn_set_sliders
        set(h_max_slider, 'callback', []);
        set(h_range_slider, 'callback', []);
        
        range_slider_val = round(options.db_range);
        if ~isempty(options.max_val)
            max_slider_val = round(20 * log10(options.max_val) + 50);
        else
            max_slider_val = 50;
        end
        range_slider_val = max([range_slider_val, 0]);
        range_slider_val = min([range_slider_val, 100]);
        set(h_range_slider, 'Value', range_slider_val);
        range_str = {sprintf('Max val: %g dB', 20 * log10(options.max_val)), ...
            sprintf('Range: %g dB', options.db_range)};
        fn_resize_control_bar;
        max_slider_val = max([max_slider_val, 0]);
        max_slider_val = min([max_slider_val, 100]);
        set(h_max_slider, 'Value', max_slider_val);
        set(h_plot_objects.range_text, 'String', range_str);
        
        set(h_max_slider, 'callback', @cb_control);
        set(h_range_slider, 'callback', @cb_control);
    end

    function fn_range_change(sh)
        set(h_max_slider, 'callback', []);
        set(h_range_slider, 'callback', []);
        max_slider_val = round(get(h_max_slider, 'value'));
        range_slider_val = round(get(h_range_slider, 'value'));
        %         disp(sprintf('Range slider val: %.2f, max slider val: %.2f', range_slider_val, max_slider_val));
        switch sh
            case h_max_slider
                max_db = max_slider_val - 50;
                options.max_val = 10 ^ (max_db / 20);
            case h_range_slider
                options.db_range = range_slider_val;
        end
        fn_update_graphs;
        fn_set_sliders;
        set(h_max_slider, 'callback', @cb_control);
        set(h_range_slider, 'callback', @cb_control);
    end

    function fn_resize_control_bar
        p = getpixelposition(h_panels.range);
        setpixelposition(h_range_slider, ...
            [1, 1, p(3), round(p(4) / 2)]);
        setpixelposition(h_max_slider, ...
            [1, round(p(4) / 2) + 1, p(3), p(4) - round(p(4) / 2)]);
    end

    function h_panels = fn_create_panels
        h_panels.range = uipanel('Parent', h_panel);
        h_panels.main = uipanel('Parent', h_panel);
        h_panels.status = uipanel('Parent', h_panel);
        h_panels.plot_3d = uipanel('Parent', h_panels.main);
        h_panels.plot_2d = uipanel('Parent', h_panels.main);
%         h_panels.control = uipanel('Parent', h_panels.main);
        fn = fieldnames(h_panels);
        for ii = 1:length(fn)
            set(getfield(h_panels, fn{ii}), 'BorderType', config.plot_panel_3d.border_type, 'BackgroundColor', config.general.window_bg_color);
        end
        set(h_panels.status, 'BorderType', config.plot_panel_3d.border_type, 'BackgroundColor', config.general.window_bg_color);
%         set(h_panels.control, 'BorderType', config.plot_panel_3d.border_type, 'BackgroundColor', config.general.window_bg_color);
%         set(h_panels.control, 'Units', 'Normalized', 'Position', [0,0,1/2,1/3]);
        set(h_panels.plot_2d, 'Units', 'Normalized', 'Position', [0,0,1/3,1]);
        set(h_panels.plot_3d, 'Units', 'Normalized', 'Position', [1/3,0,2/3,1]);
    end

    function success = fn_update_data(new_data) %this handles new data being sent to the plot panel
        if ~ishandle(h_panel)
            success = 0;
            return;
        else
            success = 1;
        end;
        plot_data = new_data;
        if ~isempty(plot_data)
            %global max is always the maximum value in the data
            options.global_max = squeeze(max(max(max(abs(plot_data.f)))));
            %other limits are only set if undefined
            %             keyboard
            if options.auto_normalise
                options.norm_val = options.global_max; %value against which everything is normalised, including max_val
                options.max_val = 1; %maximum normalised value to plot (used for linear and dB scales)
            end
            if isempty(options.max_val)|isempty(options.db_range)|isempty(options.norm_val)
                %initial set of limits (for plotting) and normalisation value
                options.norm_val = options.global_max; %value against which everything is normalised, including max_val
                options.max_val = 1; %maximum normalised value to plot (used for linear and dB scales)
                options.db_range = 30; %db range for dB scale only
                fn_set_sliders;
            end;
            %sort out x and z axes to be just vectors
            if size(plot_data.x, 1) > 1 & size(plot_data.x, 2) > 1
                if plot_data.x(1,1) == plot_data.x(1,2)
                    plot_data.x = plot_data.x(:,1);
                    plot_data.z = plot_data.z(1,:);
                else
                    plot_data.x = plot_data.x(1,:);
                    plot_data.z = plot_data.z(:,1);
                end;
            end;
            plot_data.x = plot_data.x(:);
            plot_data.z = plot_data.z(:);
            %set global axis limits
            options.global_x_lim = [min(plot_data.x), max(plot_data.x)];
            options.global_y_lim = [min(plot_data.y), max(plot_data.y)];
            options.global_z_lim = [min(plot_data.z), max(plot_data.z)];
            %setup x and z limits if not already set
            if isempty(options.x_lim)|isempty(options.y_lim)|isempty(options.z_lim)
                options.x_lim = options.global_x_lim;
                options.y_lim = options.global_y_lim;
                options.z_lim = options.global_z_lim;
            end;
            if isempty(options.x_plane) | isempty(options.y_plane) | isempty(options.z_plane) | isempty(options.active_plane)
                options.x_plane = mean(options.global_x_lim);
                options.y_plane = mean(options.global_y_lim);
                options.z_plane = mean(options.global_z_lim);
                options.active_plane = 'xy';
                
            end
            
            fn_update_graphs;
            %             set(h_axes.plot_3d, 'ButtonDownFcn', @cb_button_down_main);
            drawnow;
            %             if ~isempty(time_of_last_frame)
            %                 fps = 1 / toc(time_of_last_frame);
            %                 if fps > min_fps_to_show;
            %                     fps_string = sprintf(' FPS: %.2f ', fps);
            %                 else
            %                     fps_string = sprintf(' FPS: <%.2f ', min_fps_to_show);
            %                 end;
            %             else
            %                 fps_string = '';
            %             end;
            %             time_of_last_frame = tic;
        else
            %no data - turn off display
            %             set(h_axes.main, 'Visible', 'Off');
            options.global_max = 1;
            options.norm_val = 1;
            options.max_val = 1;
            options.db_range = 40; %db range for dB scale only
        end
    end

    function fn_resize(src, evt)
        %this preserves the width of range and status to absolute values
        p = getpixelposition(h_panel);
        setpixelposition(h_panels.range, [p(3) - config.general.slider_width_pixels, config.general.status_height_pixels + 1, config.general.slider_width_pixels, p(4) - config.general.status_height_pixels]);
        setpixelposition(h_panels.status, [1, 1, p(3), config.general.status_height_pixels]);
        setpixelposition(h_panels.main, [1, config.general.status_height_pixels + 1, p(3) - config.general.slider_width_pixels, p(4) - config.general.status_height_pixels]);
        fn_resize_control_bar;
    end

    function cb_control(src, ev)
        tag = get(src, 'Tag');
        switch tag
            %             case 'view.zoomin'
            %                 fn_radio_group(h_toolbar, tag, 0);
            %                 fn_zoom;
            %             case 'zoomout'
            %                 %                 fn_radio_group(h_toolbar, tag, 0);
            %                 fn_zoom_out;
            %             case 'view.pan'
            %                 fn_radio_group(h_toolbar, tag, 0);
            %                 fn_pan;
            case 'view.rotate'
                fn_rotate;
            case 'cursor.point'
                fn_radio_group2(h_toolbar, tag, 0);
                fn_update_cursor;
            case 'cursor.region'
                fn_radio_group2(h_toolbar, tag, 0);
                fn_update_cursor;
            case 'range'
                fn_range_change(src);
                fn_update_graphs;
            case 'max_value'
                fn_range_change(src);
                fn_update_graphs;
            case 'min_value'
                fn_range_change(src);
                fn_update_graphs;
            case 'normalise'
                fn_normalise;
                fn_update_graphs;
            case 'sections'
                fn_sections;
                fn_update_graphs;
        end
    end

    function fn_sections
        if fn_get_control_status(h_toolbar, 'sections')
            options.show_sections = 1;
        else
            options.show_sections = 0;
        end
    end

    function fn_normalise
        if isempty(options.select)
            button = questdlg('Re-normalise to global maximum?','Normalisation','Yes', 'No', 'Yes');
            if strcmp(button, 'No')
                return;
            end
            %normalise to global maximum
            options.max_val = 1;
            options.norm_val = options.global_max;
            fn_update_graphs;
            return;
        end
        if size(options.select, 1) == 1
            button = questdlg('Re-normalise to spot value?','Normalisation','Yes', 'No', 'Yes');
            if strcmp(button, 'No')
                return;
            end
            %normalise to selected point
            options.max_val = 1;
            options.norm_val = abs(fn_get_value_at_point(plot_data, options.select));
        end
        if size(options.select, 1) == 3
            button = questdlg('Re-normalise to region value?','Normalisation','Peak', 'RMS', 'Cancel', 'Peak');
            if strcmp(button, 'Cancel')
                return;
            end
            %normalise to selected region
            switch button
                case 'RMS'
                    options.norm_val = abs(fn_get_rms_in_region(plot_data, options.select([1,3],:)));
                case 'Peak'
                    options.norm_val = abs(fn_get_max_in_region(plot_data, options.select([1,3],:)));
            end
            options.max_val = 1;
        end
    end

    function cb_button_down_main(src, eventdata)
        %get line represented by cursor point in SI units
        ln = get(src, 'CurrentPoint');
        ln = ln ./ (ones(2,1) * [options.x_axis_sf, options.z_axis_sf, options.x_axis_sf]);
        last_plane = options.active_plane;
        switch src
            case h_axes.plot_xy
                options.active_plane = 'xy';
            case h_axes.plot_xz
                options.active_plane = 'xz';
            case h_axes.plot_yz
                options.active_plane = 'yz';
        end
        if strcmp(last_plane, options.active_plane)
            same_plane = 1;
        else
            same_plane = 0;
        end
        %convert to point, based on intersection with active plane
        tmp1 = [options.x_plane, options.y_plane, options.z_plane];
        switch options.active_plane
            case 'xy'
                tmp2 = tmp1 + [1,0,0];
                tmp3 = tmp1 + [0,1,0];
                lim = [options.x_lim; options.y_lim];
                lim_i = [1,2];
            case 'xz'
                tmp2 = tmp1 + [1,0,0];
                tmp3 = tmp1 + [0,0,1];
                lim = [options.x_lim; options.z_lim];
                lim_i = [1,3];
            case 'yz'
                tmp2 = tmp1 + [0,1,0];
                tmp3 = tmp1 + [0,0,1];
                lim = [options.y_lim; options.z_lim];
                lim_i = [2,3];
        end
        pln = [tmp1; tmp1 + tmp2; tmp1 + tmp3];
        point = fn_line_plane_intersection(ln, pln);
        %limit to size of axes
        for ii = 1:2
            point(lim_i(ii)) = max([point(lim_i(ii)), lim(ii, 1)]);
            point(lim_i(ii)) = min([point(lim_i(ii)), lim(ii, 2)]);
        end
        switch options.active_plane
            case 'xy'
                options.x_plane = point(1);
                options.y_plane = point(2);
            case 'xz'
                options.x_plane = point(1);
                options.z_plane = point(3);
            case 'yz'
                options.y_plane = point(2);
                options.z_plane = point(3);
        end
        if fn_get_control_status(h_toolbar, 'cursor.point')
            options.select = point;
        end;
        if fn_get_control_status(h_toolbar, 'cursor.region')
            switch next_point
                case 1
                    options.select = point;
                    next_point = 2;
                case 2
                    options.select = [options.select; point];
                    next_point = 3;
                case 3
                    %in this case the only new info is the position of the
                    %new active plane
                    if ~same_plane
                        switch last_plane
                            case 'xy'
                                options.select = [options.select; ...
                                    [options.select(2,1), options.select(2,2), point(3)]];
                            case 'xz'
                                options.select = [options.select; ...
                                    [options.select(2,1), point(2), options.select(2,3)]];
                            case 'yz'
                                options.select = [options.select; ...
                                    [point(1), options.select(2,2), options.select(2,3)]];
                        end
                        next_point = 1;
                    end;
            end;
        end;
        fn_update_cursor;
        fn_update_graphs;
    end

    function h_axes = fn_create_axes
        h_axes.plot_3d = axes('Parent', h_panels.plot_3d, 'NextPlot', 'Add', 'Layer', 'Top', 'Visible', 'On', 'ButtonDownFcn', @cb_button_down_main);
        set(h_axes.plot_3d, 'DataAspectRatio', [1,1,1], 'ZDir', 'reverse');
        view(h_axes.plot_3d, 3);
        axes(h_axes.plot_3d);
        xlabel('x (mm)');
        ylabel('y (mm)');
        zlabel('z (mm)');
        if verLessThan('matlab','R2016a')
            h_light = light('Position',[5, 5, 10],'Style','infinite');
        else
            h_light = light('Position',-[5, 5, 10],'Style','infinite');
        end
        
        h_axes.plot_xy = axes('Parent', h_panels.plot_2d, 'NextPlot', 'Add', 'Layer', 'Top', 'OuterPosition', [0,2/3,1,1/3], 'Visible', 'On', 'ButtonDownFcn', @cb_button_down_main);
        set(h_axes.plot_xy, 'DataAspectRatio', [1,1,1], 'ZDir', 'reverse', 'XTick' ,[], 'YTick', []);
        view(h_axes.plot_xy, [0,0,1]);
        axes(h_axes.plot_xy);
        xlabel('x');
        ylabel('y');
        copyobj(h_light, h_axes.plot_xy);
        
        h_axes.plot_yz = axes('Parent', h_panels.plot_2d, 'NextPlot', 'Add', 'Layer', 'Top', 'OuterPosition', [0,1/3,1,1/3], 'Visible', 'On', 'ButtonDownFcn', @cb_button_down_main);
        set(h_axes.plot_yz, 'DataAspectRatio', [1,1,1], 'ZDir', 'reverse', 'ZTick' ,[], 'YTick', []);
        view(h_axes.plot_yz, [-1,0,0]);
        axes(h_axes.plot_yz);
        zlabel('z');
        ylabel('y');
        copyobj(h_light, h_axes.plot_yz);
        
        h_axes.plot_xz = axes('Parent', h_panels.plot_2d, 'NextPlot', 'Add', 'Layer', 'Top', 'OuterPosition', [0,0,1,1/3], 'Visible', 'On', 'ButtonDownFcn', @cb_button_down_main);
        set(h_axes.plot_xz, 'DataAspectRatio', [1,1,1], 'ZDir', 'reverse', 'XTick' ,[], 'ZTick', []);
        view(h_axes.plot_xz, [0,-1,0]);
        axes(h_axes.plot_xz);
        xlabel('x');
        zlabel('z');
        copyobj(h_light, h_axes.plot_xz);
    end

    function h_plot_objects = fn_create_plot_objects
        %main isosurface
        h_plot_objects.isosurface = patch('Parent', h_axes.plot_3d, 'Visible', 'Off', 'HitTest', 'off', 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', config.plot_panel_3d.isosurf_alpha, 'Tag', 'isosurface','CDataMapping','direct');
        %plot end caps on 3D graph
        h_plot_objects.isocap = patch('Parent', h_axes.plot_3d, 'Visible', 'Off', 'HitTest', 'off', 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', config.plot_panel_3d.isosurf_alpha, 'Tag', 'isocap','CDataMapping','direct');
        %section planes in 3D graph
        h_plot_objects.planes_3d = patch('Parent', h_axes.plot_3d, 'Visible', 'Off', 'HitTest', 'off', 'FaceColor', config.plot_panel_3d.plane_color, 'EdgeColor', config.plot_panel_3d.plane_color, 'Vertices', zeros(12,3), 'Faces', [1,2,3,4; 5,6,7,8; 9,10,11,12], 'FaceAlpha', config.plot_panel_3d.plane_alpha, 'Tag', 'planes');
        %active plane border in 3D graph
        h_plot_objects.active_plane_3d = line('Parent', h_axes.plot_3d, 'HitTest', 'off', 'XData',zeros(5,1),'YData',zeros(5,1),'ZData',zeros(5,1),'Color', config.plot_panel_3d.active_color, 'Visible', 'Off', 'Tag', 'planes');
        %current point
        h_plot_objects.active_point = line('Parent', h_axes.plot_3d, 'HitTest', 'off', 'XData',zeros(1,1),'YData',zeros(1,1),'ZData',zeros(1,1),'Color', config.plot_panel_3d.selection_color, 'Marker','.', 'Visible', 'Off', 'Tag', 'selection');
        %plane intersection lines
        h_plot_objects.intersection_lines = line('Parent', h_axes.plot_3d, 'HitTest', 'off', 'XData',zeros(1,1),'YData',zeros(1,1),'ZData',zeros(1,1),'Color', config.plot_panel_3d.selection_color, 'LineStyle',':', 'Visible', 'Off', 'Tag', 'selection');
        %selection volume
        h_plot_objects.selection = patch('Parent', h_axes.plot_3d, 'Visible', 'Off', 'HitTest', 'off', 'FaceColor', config.plot_panel_3d.selection_color, 'EdgeColor', config.plot_panel_3d.selection_color', 'Vertices', zeros(8,3), 'Faces', [1,2,3,4; 5,6,7,8], 'FaceAlpha', config.plot_panel_3d.selection_alpha, 'Tag', 'selection');
        %plane surface plots
        h_plot_objects.plane1 = surf('Parent', h_axes.plot_yz, 'Visible', 'Off', 'HitTest', 'off', 'FaceColor', 'interp', 'LineStyle', 'none', 'FaceLighting', 'None');
        h_plot_objects.plane2 = surf('Parent', h_axes.plot_xz, 'Visible', 'Off', 'HitTest', 'off', 'FaceColor', 'interp', 'LineStyle', 'none', 'FaceLighting', 'None');
        h_plot_objects.plane3 = surf('Parent', h_axes.plot_xy, 'Visible', 'Off', 'HitTest', 'off', 'FaceColor', 'interp', 'LineStyle', 'none', 'FaceLighting', 'None');
        
        h_plot_objects.selection_text = uicontrol(h_panels.status, 'Style', 'Text', 'String', {'1','2','3'}, 'HorizontalAlignment', 'Left', 'Units', 'Normalized', 'Position', [0.02, 0, 0.5, 1], 'BackgroundColor', config.general.window_bg_color);
        h_plot_objects.range_text = uicontrol(h_panels.status, 'Style', 'Text', 'String', {'1','2','3'}, 'HorizontalAlignment', 'Right', 'Units', 'Normalized', 'Position', [0.5, 0, 0.48, 1], 'BackgroundColor', config.general.window_bg_color);
        
    end

    function fn_update_graphs
        fn_set_options_to_match_toolbar_btns;
        %         fn_set_sliders;
        if isempty(plot_data)% | ~data_has_been_plotted
            %             h_plot_objects.selection_text = uicontrol(h_panels.main, 'Style', 'Text', 'String', '', 'HorizontalAlignment', 'Center', 'Units', 'Normalized', 'Position', [0, 0.9, 1, 0.1], 'BackgroundColor', config.general.window_bg_color);
            %should clear graphs here - e.g. plot crosses
            return;
        end
        
        %update main graph
        [plot_val, limits, options.scale_mode] = fn_convert_to_plot_val(plot_data.f, 'mod', 'log', options.db_range, options.max_val, options.norm_val);
        
        %create isosurface
        [z_arr]=repmat(plot_data.z * options.z_axis_sf,[1 length(plot_data.x) length(plot_data.y)]);
        z_arr=permute(z_arr, [3 2 1]);
        %[x_arr y_arr z_arr]=meshgrid(plot_data.x * options.x_axis_sf,plot_data.y * options.y_axis_sf,plot_data.z * options.z_axis_sf);
        %[f, v] = isosurface(plot_data.x * options.x_axis_sf, plot_data.y * options.y_axis_sf, plot_data.z * options.z_axis_sf, plot_val, -options.db_range, 'no share');
        %[f, v, color_vals] = isosurface(x_arr, y_arr, z_arr, plot_val, -options.db_range, z_arr, 'no share');
        [f, v, color_vals] = isosurface(plot_data.x * options.x_axis_sf, plot_data.y * options.y_axis_sf, plot_data.z * options.z_axis_sf, plot_val, -options.db_range, z_arr, 'no share');
        
        [fcap, vcap, capcolor_vals] = isocaps(plot_data.x * options.x_axis_sf, plot_data.y * options.y_axis_sf, plot_data.z * options.z_axis_sf, plot_val, -options.db_range);
        
        
        %set up objects accordingly in 3D view
        set(h_plot_objects.isosurface, ...
            'Faces', f, ...
            'Vertices', v, ...
            'FaceVertexCData', color_vals./max(plot_data.z * options.z_axis_sf).*64, ...
            'Visible', 'On');
        
        set(h_plot_objects.isocap, ...
            'Faces', fcap, ...
            'Vertices', vcap, ...
            'FaceVertexCData', ((capcolor_vals+options.db_range)./options.db_range).*64, ...
            'Visible', 'On');
        
        if ~options.show_sections
            
            set(h_plot_objects.plane1, 'Visible', 'Off');
            set(h_plot_objects.plane2, 'Visible', 'Off');
            set(h_plot_objects.plane3, 'Visible', 'Off');
            
            %copy to other views
            delete(findobj(h_axes.plot_xy, 'Tag', 'isosurface'));
            delete(findobj(h_axes.plot_xz, 'Tag', 'isosurface'));
            delete(findobj(h_axes.plot_yz, 'Tag', 'isosurface'));
            copyobj(h_plot_objects.isosurface, h_axes.plot_xy);
            copyobj(h_plot_objects.isosurface, h_axes.plot_xz);
            copyobj(h_plot_objects.isosurface, h_axes.plot_yz);
            
            delete(findobj(h_axes.plot_xy, 'Tag', 'isocap'));
            delete(findobj(h_axes.plot_xz, 'Tag', 'isocap'));
            delete(findobj(h_axes.plot_yz, 'Tag', 'isocap'));
            copyobj(h_plot_objects.isocap, h_axes.plot_xy);
            copyobj(h_plot_objects.isocap, h_axes.plot_xz);
            copyobj(h_plot_objects.isocap, h_axes.plot_yz);
        else
            
            delete(findobj(h_axes.plot_xy, 'Tag', 'isosurface'));
            delete(findobj(h_axes.plot_xz, 'Tag', 'isosurface'));
            delete(findobj(h_axes.plot_yz, 'Tag', 'isosurface'));
            delete(findobj(h_axes.plot_xy, 'Tag', 'isocap'));
            delete(findobj(h_axes.plot_xz, 'Tag', 'isocap'));
            delete(findobj(h_axes.plot_yz, 'Tag', 'isocap'));
            
            set(h_axes.plot_xz,'CLim',[-(options.db_range*2) 0],'CLimMode','manual');
            set(h_axes.plot_yz,'CLim',[-(options.db_range*2) 0],'CLimMode','manual');
            set(h_axes.plot_xy,'CLim',[-(options.db_range*2) 0],'CLimMode','manual');
            
            %calculate the view planes
            x=plot_data.x * options.x_axis_sf;%
            y=plot_data.y * options.y_axis_sf;%
            z=plot_data.z * options.z_axis_sf;%

            x_ind=round((options.x_plane-plot_data.x(1))./(plot_data.x(2)-plot_data.x(1))+1);
            y_ind=round((options.y_plane-plot_data.y(1))./(plot_data.y(2)-plot_data.y(1))+1);
            z_ind=round((options.z_plane-plot_data.z(1))./(plot_data.z(2)-plot_data.z(1))+1);
            
            if x_ind>length(x)
                x_ind=length(x);
            end
            if y_ind>length(y)
                y_ind=length(y);
            end
            if z_ind>length(z)
                z_ind=length(z);
            end
            
            if x_ind<1
                x_ind=1;
            end
            if y_ind<1
                y_ind=1;
            end
            if z_ind<1
                z_ind=1;
            end
            
            [yi,zi] = meshgrid(y,z);
            xi = ones(size(yi)) * mean(x);
            vi = squeeze(plot_val(:,x_ind,:))';
            
            set(h_plot_objects.plane1, ...
                'XData', xi, ...
                'YData', yi, ...
                'zData', zi, ...
                'CData', vi, ...
                'Visible', 'On');
                        
            
            [xi,zi] = meshgrid(x,z);
            yi = ones(size(xi)) * mean(y);
            vi = squeeze(plot_val(y_ind,:,:))';
            
            set(h_plot_objects.plane2, ...
                'XData', xi, ...
                'YData', yi, ...
                'zData', zi, ...
                'CData', vi, ...
                'Visible', 'On');
            
            [xi,yi] = meshgrid(x,y);
            zi = ones(size(xi)) * mean(z);
            vi = squeeze(plot_val(:,:,z_ind));
            
            set(h_plot_objects.plane3, ...
                'XData', xi, ...
                'YData', yi, ...
                'zData', zi, ...
                'CData', vi, ...
                'Visible', 'On');
            
            
        end
        %         if ~data_has_been_plotted
        axis(h_axes.plot_3d, ...
            [options.global_x_lim * options.x_axis_sf, ...
            options.global_y_lim * options.y_axis_sf, ...
            options.global_z_lim * options.z_axis_sf]);
        axis(h_axes.plot_xy, ...
            [options.global_x_lim * options.x_axis_sf, ...
            options.global_y_lim * options.y_axis_sf, ...
            options.global_z_lim * options.z_axis_sf]);
        axis(h_axes.plot_xz, ...
            [options.global_x_lim * options.x_axis_sf, ...
            options.global_y_lim * options.y_axis_sf, ...
            options.global_z_lim * options.z_axis_sf]);
        axis(h_axes.plot_yz, ...
            [options.global_x_lim * options.x_axis_sf, ...
            options.global_y_lim * options.y_axis_sf, ...
            options.global_z_lim * options.z_axis_sf]);
        %         end
        %         data_has_been_plotted = 1;
        
        fn_update_cursor;
        
        fn_set_sliders;
        
    end

    function fn_update_cursor
        if ~fn_get_control_status(h_toolbar, 'cursor.point') & ~fn_get_control_status(h_toolbar, 'cursor.region')
            options.select = [];
        end
        if fn_get_control_status(h_toolbar, 'cursor.point')
            if size(options.select, 1)
                val_str = sprintf('%.1f dB', 20*log10(abs(fn_get_value_at_point(plot_data, options.select)) / options.norm_val));
                pos_str = sprintf('(%.2f, %.2f, %.2f)', options.select(1,1) * options.x_axis_sf, options.select(1,2) * options.y_axis_sf, options.select(1,3) * options.z_axis_sf);
                pointer_str = {pos_str, val_str};
            end
        end
        if fn_get_control_status(h_toolbar, 'cursor.region')
            if size(options.select, 1) > 0
                pos_str = sprintf('(%.2f, %.2f, %.2f)', options.select(1,1) * options.x_axis_sf, options.select(1,2) * options.y_axis_sf, options.select(1,3) * options.z_axis_sf);
                if size(options.select, 1) < 3
                    val_str = sprintf('%.1f dB', 20*log10(abs(fn_get_value_at_point(plot_data, options.select(1, :))) / options.norm_val));
                    pointer_str = {pos_str, val_str};
                else
                    val_str = [sprintf('%.1f dB (max); ', 20*log10(abs(fn_get_max_in_region(plot_data, options.select([1,3],:))) / options.norm_val)), ...
                        sprintf('%.1f dB (rms)', 20*log10(abs(fn_get_rms_in_region(plot_data, options.select([1,3],:))) / options.norm_val))];
                    pos_str = [pos_str, ' to ', ...
                        sprintf('(%.2f, %.2f, %.2f)', options.select(3,1) * options.x_axis_sf, options.select(3,2) * options.y_axis_sf, options.select(3,3) * options.z_axis_sf)];
                end;
                pointer_str = {pos_str, val_str};
            end
        end
        
        %graphical part
        cut_plane_vertices = [...
            options.global_x_lim(1) * options.x_axis_sf, options.global_y_lim(1) * options.y_axis_sf, options.z_plane * options.z_axis_sf; ...
            options.global_x_lim(2) * options.x_axis_sf, options.global_y_lim(1) * options.y_axis_sf, options.z_plane * options.z_axis_sf; ...
            options.global_x_lim(2) * options.x_axis_sf, options.global_y_lim(2) * options.y_axis_sf, options.z_plane * options.z_axis_sf; ...
            options.global_x_lim(1) * options.x_axis_sf, options.global_y_lim(2) * options.y_axis_sf, options.z_plane * options.z_axis_sf; ...
            
            options.x_plane * options.x_axis_sf, options.global_y_lim(1) * options.y_axis_sf, options.global_z_lim(1) * options.z_axis_sf; ...
            options.x_plane * options.x_axis_sf, options.global_y_lim(2) * options.y_axis_sf, options.global_z_lim(1) * options.z_axis_sf; ...
            options.x_plane * options.x_axis_sf, options.global_y_lim(2) * options.y_axis_sf, options.global_z_lim(2) * options.z_axis_sf; ...
            options.x_plane * options.x_axis_sf, options.global_y_lim(1) * options.y_axis_sf, options.global_z_lim(2) * options.z_axis_sf; ...
            
            options.global_x_lim(1) * options.x_axis_sf, options.y_plane * options.y_axis_sf, options.global_z_lim(1) * options.z_axis_sf; ...
            options.global_x_lim(2) * options.x_axis_sf, options.y_plane * options.y_axis_sf, options.global_z_lim(1) * options.z_axis_sf; ...
            options.global_x_lim(2) * options.x_axis_sf, options.y_plane * options.y_axis_sf, options.global_z_lim(2) * options.z_axis_sf; ...
            options.global_x_lim(1) * options.x_axis_sf, options.y_plane * options.y_axis_sf, options.global_z_lim(2) * options.z_axis_sf];
        point_vertices = [cut_plane_vertices(5,1), cut_plane_vertices(9,2), cut_plane_vertices(1,3)];
%         line_vertices_x = [cut_plane_vertices([1,5,2],1); NaN; cut_plane_vertices([5,5,5],1); NaN; cut_plane_vertices([5,5,5],1)] * options.x_axis_sf;
%         line_vertices_y = [cut_plane_vertices([1,1,1],2); NaN; cut_plane_vertices([5,1,6],2); NaN; cut_plane_vertices([1,1,1],2)] * options.y_axis_sf;
%         line_vertices_z = [cut_plane_vertices([9,9,9],3); NaN; cut_plane_vertices([9,9,9],3); NaN; cut_plane_vertices([5,9,8],3)] * options.z_axis_sf;
        switch options.active_plane
            case 'xy'
                cut_boarder = cut_plane_vertices([1:4, 1], :);
            case 'yz'
                cut_boarder = cut_plane_vertices([5:8, 5], :);
            case 'xz'
                cut_boarder = cut_plane_vertices([9:12, 9], :);
        end
        
        set(h_plot_objects.planes_3d, ...
            'Vertices', cut_plane_vertices, ...
            'Visible', 'On');
        set(h_plot_objects.active_plane_3d, 'XData', cut_boarder(:,1), 'YData', cut_boarder(:,2), 'ZData', cut_boarder(:,3), 'Visible', 'On');
        set(h_plot_objects.active_point, 'XData', point_vertices(:,1), 'YData', point_vertices(:,2), 'ZData', point_vertices(:,3), 'Visible', 'On');
%         set(h_plot_objects.intersection_lines, 'XData', line_vertices_x, 'YData', line_vertices_y, 'ZData', line_vertices_z, 'Visible', 'On');
        switch size(options.select, 1)
            case 1
                set(h_plot_objects.selection, 'Visible', 'Off');
            case 2
                switch options.active_plane
                    case 'xy'
                        select_vertices = [ ...
                            options.select(1,1), options.select(1,2), options.select(1,3); ...
                            options.select(2,1), options.select(1,2), options.select(1,3); ...
                            options.select(2,1), options.select(2,2), options.select(1,3); ...
                            options.select(1,1), options.select(2,2), options.select(1,3)] ...
                            .* (ones(4,1) * [options.x_axis_sf, options.y_axis_sf, options.z_axis_sf]);
                    case 'yz'
                        select_vertices = [ ...
                            options.select(1,1), options.select(1,2), options.select(1,3); ...
                            options.select(1,1), options.select(2,2), options.select(1,3); ...
                            options.select(1,1), options.select(2,2), options.select(2,3); ...
                            options.select(1,1), options.select(1,2), options.select(2,3)] ...
                            .* (ones(4,1) * [options.x_axis_sf, options.y_axis_sf, options.z_axis_sf]);
                    case 'xz'
                        select_vertices = [ ...
                            options.select(1,1), options.select(1,2), options.select(1,3); ...
                            options.select(2,1), options.select(1,2), options.select(1,3); ...
                            options.select(2,1), options.select(1,2), options.select(2,3); ...
                            options.select(1,1), options.select(1,2), options.select(2,3)] ...
                            .* (ones(4,1) * [options.x_axis_sf, options.y_axis_sf, options.z_axis_sf]);
                end
                select_faces = [1,2,3,4];
                set(h_plot_objects.selection, 'Visible', 'On', 'Vertices', select_vertices, 'Faces', select_faces);
            case 3
                select_vertices = [ ...
                    options.select(1,1), options.select(1,2), options.select(1,3); ...
                    options.select(3,1), options.select(1,2), options.select(1,3); ...
                    options.select(3,1), options.select(3,2), options.select(1,3); ...
                    options.select(1,1), options.select(3,2), options.select(1,3); ...
                    options.select(1,1), options.select(1,2), options.select(3,3); ...
                    options.select(3,1), options.select(1,2), options.select(3,3); ...
                    options.select(3,1), options.select(3,2), options.select(3,3); ...
                    options.select(1,1), options.select(3,2), options.select(3,3)] ...
                    .* (ones(8,1) * [options.x_axis_sf, options.y_axis_sf, options.z_axis_sf]);
                select_faces = [1,2,3,4; 1,2,6,5; 1,4,8,5; 2,3,7,6; 4,3,7,8; 5,6,7,8];
                set(h_plot_objects.selection, 'Visible', 'On', 'Vertices', select_vertices, 'Faces', select_faces);
        end
        
        %copy from 3D to 2D views (currently all objects, but this could be more
        %selective!)
        %copy to other views
        delete(findobj(h_axes.plot_xy, 'Tag', 'planes'));
        delete(findobj(h_axes.plot_xz, 'Tag', 'planes'));
        delete(findobj(h_axes.plot_yz, 'Tag', 'planes'));
        h = findobj(h_axes.plot_3d, 'Tag', 'planes');
        copyobj(h, h_axes.plot_xy);
        copyobj(h, h_axes.plot_xz);
        copyobj(h, h_axes.plot_yz);
        if options.show_sections
            hh = findobj(h_axes.plot_xy, 'Tag', 'planes', 'Type', 'patch');
            set(hh, 'FaceAlpha', 0);
            hh = findobj(h_axes.plot_xz, 'Tag', 'planes', 'Type', 'patch');
            set(hh, 'FaceAlpha', 0);
            hh = findobj(h_axes.plot_yz, 'Tag', 'planes', 'Type', 'patch');
            set(hh, 'FaceAlpha', 0);
        end
        
        %Update strings
        set(h_plot_objects.selection_text, 'String', pointer_str);
        set(h_plot_objects.range_text, 'String', range_str);
    end

    function fn_colorbar
        if fn_get_control_status(h_toolbar, 'colorbar')
            colorbar('EastOutside', 'peer', h_axes.main);
            options.show_colorbar = 1;
        else
            colorbar('Off', 'peer', h_axes.main);
            options.show_colorbar = 0;
        end;
    end

%     function fn_zoom_out
%         options.x_lim = options.global_x_lim;
%         options.z_lim = options.global_z_lim;
%         axis(h_axes.main, [options.global_x_lim * options.x_axis_sf, options.global_z_lim * options.z_axis_sf]);
%         fn_update_graphs;
%     end
%
%
%     function fn_zoom
%         if fn_get_control_status(h_toolbar, 'view.zoomin')
%             set(h_zoom, 'Enable','On');
%             set(h_zoom, 'Direction','In');
%         else
%             set(h_zoom, 'Enable', 'Off');
%         end
%     end
%
%     function fn_pan
%         if fn_get_control_status(h_toolbar, 'view.pan')
%             pan on;
%         else
%             pan off;
%         end;
%     end

    function fn_rotate
        if fn_get_control_status(h_toolbar, 'view.rotate')
            rotate3d(h_axes.plot_3d, 'on');
        else
            rotate3d(h_axes.plot_3d, 'off');
        end;
    end
end


function [plot_val, limits, new_scale_mode] = fn_convert_to_plot_val(val, plotwhat, scale_mode, db_range, max_val, norm_val)
val = val / norm_val;
switch plotwhat
    case 'mod'
        plot_val = abs(val);
        limits = [0, max_val];
    case 'real'
        scale_mode = 'linear';
        plot_val = real(val);
        limits = [-max_val, max_val];
    case 'mod_real'
        plot_val = abs(real(val));
        limits = [0, max_val];
    case 'imag'
        scale_mode = 'linear';
        plot_val = imag(val);
        limits = [-max_val, max_val];
    case 'mod_imag'
        plot_val = abs(imag(val));
        limits = [0, max_val];
    case 'arg'
        scale_mode = 'linear';
        plot_val = angle(val) * 180 / pi;
        limits = [-180, 180];
        norm_val = 1;
end;
switch scale_mode
    case 'log'
        plot_val = 20 * log10(plot_val);
        limits = 20 * log10(max_val) + [-db_range, 0];
    case 'linear'
        %do nothing!
end;
new_scale_mode = scale_mode;
end

function fn_mutually_exclusive_handler(handles, tag, src)
fn = fieldnames(handles);
kk = [];
for ii = 1:length(fn)
    if strcmpi(get(getfield(handles, fn{ii}), 'tag'), tag)
        kk = [kk, ii];
    end;
end;
tmp = get(getfield(handles, fn{kk(1)}), 'callback');
%disable callbacks
for ii = 1:length(kk)
    set(getfield(handles, fn{kk(ii)}), 'callback', []);
end;
%unclick the ones not clicked on
for ii = 1:length(kk)
    if getfield(handles, fn{kk(ii)}) ~= src
        set(getfield(handles, fn{kk(ii)}), 'value', 0);
    end
end
%enable callbacks
for ii = 1:length(kk)
    set(getfield(handles, fn{kk(ii)}), 'callback', tmp);
end;
end

function res = fn_get_control_status(h_root, tag)
h = findall(h_root, 'Tag', tag);
if isempty(h)
    res = 0;
    return;
end
if strcmpi(get(h, 'State'), 'On')
    res = 1;
else
    res = 0;
end
end

function fn_set_control_status(h_root, tag, val)
h = findall(h_root, 'Tag', tag);
if isempty(h)
    return;
end
if val
    set(h, 'State', 'On');
else
    set(h, 'State', 'Off');
end;
end

function val = fn_get_value_at_point(data, pt)
i1 = interp1(data.x, [1:length(data.x)], pt(1), 'nearest');
i2 = interp1(data.y, [1:length(data.y)], pt(2), 'nearest');
i3 = interp1(data.z, [1:length(data.z)], pt(3), 'nearest');
val = data.f(i2, i1, i3);
end

function val = fn_get_rms_in_region(data, reg)
i1 = find(data.x >= min(reg(:, 1)) & data.x <= max(reg(:, 1)));
i2 = find(data.y >= min(reg(:, 2)) & data.y <= max(reg(:, 2)));
i3 = find(data.z >= min(reg(:, 3)) & data.z <= max(reg(:, 3)));
val = data.f(i2, i1, i3);
val = sqrt(mean(val(:) .* conj(val(:))));
end

function val = fn_get_max_in_region(data, reg)
i1 = find(data.x >= min(reg(:, 1)) & data.x <= max(reg(:, 1)));
i2 = find(data.y >= min(reg(:, 2)) & data.y <= max(reg(:, 2)));
i3 = find(data.z >= min(reg(:, 3)) & data.z <= max(reg(:, 3)));
val = data.f(i2, i1, i3);
val = max(abs(val(:)));
end