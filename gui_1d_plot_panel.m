function [h_fn_update_data, h_fn_get_options, h_fn_set_options] = gui_1d_plot_panel(h_panel, h_toolbar)
default_options.custom_button = [];
default_options.plotwhat = 'mod';
default_options.scale_mode = 'log';
default_options.max_val = [];
default_options.db_range = [];
default_options.global_max = [];
default_options.norm_val = [];
default_options.x_lim = [];
default_options.z_lim = [];
default_options.x_label = '';
default_options.z_label = '';
default_options.select = [];
default_options.x_axis_sf = 1e6;
default_options.cursor_type = 'none';
default_options.auto_normalise = 1;

default_options.clicked_custom_button_index = [];
options = [];
options = fn_set_default_fields(options, default_options);

data_has_been_plotted = 0;

%load configuration data
config = fn_get_config;

panel_colour = config.general.window_bg_color;
panel_border_type = config.plot_panel_2d.border_type;
button_size_pixels = config.general.button_height_pixels;
status_height_pixels = config.general.status_height_pixels;
slider_width_pixels = config.general.slider_width_pixels;

next_point = 1;
time_of_last_frame = [];
pointer_str = '';
range_str = '';
reset_x_lim = 0;
reset_y_lim = 0;

max_lines = 5;
line_colors = 'rgbmc';

orig_data = [];
plot_data = [];

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
fn_update_layout;
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
        fn_set_custom_buttons;
        fn_set_controls_to_match_options;
        fn_update_layout;
        fn_update_graphs;
    end

    function fn_set_controls_to_match_options %sets controls to match options - called when panel is created
        switch options.scale_mode
            case 'log'
                fn_set_control_status(h_toolbar, 'scale.log', 1);
                fn_set_control_status(h_toolbar, 'scale.linear', 0);
            case 'linear'
                fn_set_control_status(h_toolbar, 'scale.log', 0);
                fn_set_control_status(h_toolbar, 'scale.linear', 1);
        end
        switch options.plotwhat
            case 'mod'
                fn_set_control_status(h_toolbar, 'plotwhat.mod', 1);
                fn_set_control_status(h_toolbar, 'plotwhat.real', 0);
                fn_set_control_status(h_toolbar, 'plotwhat.arg', 0);
            case 'real'
                fn_set_control_status(h_toolbar, 'plotwhat.mod', 0);
                fn_set_control_status(h_toolbar, 'plotwhat.real', 1);
                fn_set_control_status(h_toolbar, 'plotwhat.arg', 0);
            case 'arg'
                fn_set_control_status(h_toolbar, 'plotwhat.mod', 0);
                fn_set_control_status(h_toolbar, 'plotwhat.real', 0);
                fn_set_control_status(h_toolbar, 'plotwhat.arg', 1);
        end
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
        for ii = 1:max_lines
            fn_set_control_status(h_toolbar, sprintf('show_line_%i', ii), 1);
        end
        fn_set_sliders;
    end

    function fn_create_controls
        %buttons in toolbar
        h_toolbar_btns(1).separator = 1;
        h_toolbar_btns(1).tag = 'view.zoomin';
        h_toolbar_btns(1).tooltip = 'Zoom in';
        h_toolbar_btns(1).icon = fn_get_cdata_for_named_icon(icons, 'Exploration.ZoomIn');
        h_toolbar_btns(1).type = 'toggle';
        
        h_toolbar_btns(2).tag = 'zoomout';
        h_toolbar_btns(2).tooltip = 'Zoom out';
        h_toolbar_btns(2).icon = fn_get_cdata_for_named_icon(icons, 'Exploration.ZoomOut');
        h_toolbar_btns(2).type = 'pushtool';
        
        h_toolbar_btns(3).tag = 'view.pan';
        h_toolbar_btns(3).tooltip = 'Pan';
        h_toolbar_btns(3).icon = fn_get_cdata_for_named_icon(icons, 'Exploration.Pan');
        h_toolbar_btns(3).type = 'toggle';
        
        h_toolbar_btns(4).separator = 1;
        h_toolbar_btns(4).tag = 'cursor.point';
        h_toolbar_btns(4).tooltip = 'Select point';
        h_toolbar_btns(4).icon = fn_get_cdata_for_named_icon(icons, 'point');
        h_toolbar_btns(4).type = 'toggle';
        
        h_toolbar_btns(5).tag = 'cursor.region';
        h_toolbar_btns(5).tooltip = 'Select region';
        h_toolbar_btns(5).icon = fn_get_cdata_for_named_icon(icons, 'region');
        h_toolbar_btns(5).type = 'toggle';
        
        h_toolbar_btns(6).separator = 1;
        h_toolbar_btns(6).tag = 'scale.linear';
        h_toolbar_btns(6).tooltip = 'Linear scale';
        h_toolbar_btns(6).icon = fn_get_cdata_for_named_icon(icons, 'linear');
        h_toolbar_btns(6).type = 'toggle';
        
        h_toolbar_btns(7).tag = 'scale.log';
        h_toolbar_btns(7).tooltip = 'Log scale';
        h_toolbar_btns(7).icon = fn_get_cdata_for_named_icon(icons, 'log');
        h_toolbar_btns(7).type = 'toggle';
        
        h_toolbar_btns(8).separator = 1;
        h_toolbar_btns(8).tag = 'plotwhat.real';
        h_toolbar_btns(8).tooltip = 'Plot real part';
        h_toolbar_btns(8).icon = fn_get_cdata_for_named_icon(icons, 'real');
        h_toolbar_btns(8).type = 'toggle';
        
        h_toolbar_btns(9).tag = 'plotwhat.mod';
        h_toolbar_btns(9).tooltip = 'Plot modulus';
        h_toolbar_btns(9).icon = fn_get_cdata_for_named_icon(icons, 'modulus');
        h_toolbar_btns(9).type = 'toggle';
        
        h_toolbar_btns(10).tag = 'plotwhat.arg';
        h_toolbar_btns(10).tooltip = 'Plot argument';
        h_toolbar_btns(10).icon = fn_get_cdata_for_named_icon(icons, 'argument');
        h_toolbar_btns(10).type = 'toggle';
        
        h_toolbar_btns(11).separator = 1;
        h_toolbar_btns(11).tag = 'normalise';
        h_toolbar_btns(11).tooltip = 'Normalise';
        h_toolbar_btns(11).icon = fn_get_cdata_for_named_icon(icons, 'normalise');
        h_toolbar_btns(11).type = 'pushtool';
        
        h_toolbar_btns(12).tag = 'auto_normalise';
        h_toolbar_btns(12).tooltip = 'Auto normalise';
        h_toolbar_btns(12).icon = fn_get_cdata_for_named_icon(icons, 'auto_normalise');
        h_toolbar_btns(12).type = 'toggle';
        
        for ii = 1: max_lines
            jj = length(h_toolbar_btns) + 1;
            str = sprintf('%i', ii);
            h_toolbar_btns(jj).tag = ['show_line_', str];
            h_toolbar_btns(jj).tooltip = ['Show line ', str];
            h_toolbar_btns(jj).icon = repmat(reshape(rem(floor((strfind('kbgcrmyw', line_colors(ii)) - 1) * [0.25 0.5 1]), 2), [1,1,3]), [16,16,1]);
            h_toolbar_btns(jj).type = 'toggle';
            if ii == 1
                h_toolbar_btns(jj).separator = 1;
            end
        end
        
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
        h_max_slider = uicontrol(h_panels.control, ...
            'Style', 'slider', 'String', '37', ...
            'Callback', @cb_control, 'Min', 0, 'Max', 100, 'Value', 40, ...
            'Tag', 'max_value');
        h_range_slider = uicontrol(h_panels.control, ...
            'Style', 'slider', 'String', '37', ...
            'Callback', @cb_control, 'Min', 0, 'Max', 100, 'Value', 40, ...
            'Tag', 'range');
    end

    function fn_set_custom_buttons
        %add custom buttons up the RH side (from bottom)
        ch = findall(h_panels.control, 'tag', 'custom');
        if ~isempty(ch)
            delete(ch);
        end
        %set the new controls
        kk = length(h_toolbar_btns) + 1;
        if ~isempty(options.custom_button)
            for ii = 1:length(options.custom_button)
                h_custom_button(kk).handle = uicontrol(h_panels.control, ...
                    'String', options.custom_button(ii).string, ...
                    'Style', 'pushbutton', ...
                    'Callback', @cb_control, ...
                    'Tag', 'custom');
                kk = kk + 1;
            end;
        end;
        fn_resize_control_bar;
    end

    function fn_set_sliders
        set(h_max_slider, 'callback', []);
        set(h_range_slider, 'callback', []);
        
        switch options.scale_mode
            case 'log'
                set(h_range_slider, 'Visible' , 'On');
                if ~isempty(options.db_range)
                    range_slider_val = round(options.db_range);
                else
                    range_slider_val = 50;
                end
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
            case 'linear'
                set(h_range_slider, 'Visible' , 'Off');
                if ~isempty(options.max_val)
                    max_slider_val = round(100 - 50 * options.max_val);
                end
                range_str = sprintf('%i %% full scale', options.max_val * 100);
        end
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
        switch options.scale_mode
            case 'log'
                switch sh
                    case h_max_slider
                        max_db = max_slider_val - 50;
                        options.max_val = 10 ^ (max_db / 20);
                    case h_range_slider
                        options.db_range = range_slider_val;
                end
            case 'linear'
                options.max_val = (100 - max_slider_val) / 50;
                options.max_val = max([options.max_val, 0.01]); %avoid range problem!
        end
        fn_update_graphs;
        fn_set_sliders;
        set(h_max_slider, 'callback', @cb_control);
        set(h_range_slider, 'callback', @cb_control);
    end

    function fn_resize_control_bar
        p = getpixelposition(h_panels.control);
        if strcmpi(get(h_range_slider, 'Visible'), 'On')
            setpixelposition(h_range_slider, ...
                [1, 1, p(3), round(p(4) / 2)]);
            setpixelposition(h_max_slider, ...
                [1, round(p(4) / 2) + 1, p(3), p(4) - round(p(4) / 2)]);
        else
            setpixelposition(h_max_slider, ...
                [1, 1, p(3), p(4)]);
        end
    end

    function h_plot_objects = fn_create_plot_objects
        h_plot_objects.z_crosshair = plot(h_axes.main, [0,0], [1,1], 'Visible', 'Off', 'LineStyle', ':', 'Color', 'k', 'HitTest', 'off');
        h_plot_objects.region = plot(h_axes.main, [0,0], [1,1], 'Visible', 'Off', 'LineStyle', ':', 'Marker', 's', 'Color', 'k', 'HitTest', 'off');
        
        h_plot_objects.selection_text = uicontrol(h_panels.status, 'Style', 'Text', 'String', {'1','2','3'}, 'HorizontalAlignment', 'Left', 'Units', 'Normalized', 'Position', [0.02, 0, 0.5, 1], 'BackgroundColor', panel_colour);
        h_plot_objects.range_text = uicontrol(h_panels.status, 'Style', 'Text', 'String', {'1','2','3'}, 'HorizontalAlignment', 'Right', 'Units', 'Normalized', 'Position', [0.5, 0, 0.48, 1], 'BackgroundColor', panel_colour);
    end

    function h_axes = fn_create_axes
        h_axes.main = axes('Parent', h_panels.main, 'NextPlot', 'Add', 'Layer', 'Top');
    end

    function h_panels = fn_create_panels
        h_panels.plot = uipanel('Parent', h_panel);
        h_panels.main = uipanel('Parent', h_panels.plot, ...
            'BorderType',  config.plot_panel_2d.graph_panel.border_type, ...
            'BackgroundColor', config.general.window_bg_color);
        h_panels.status = uipanel('Parent', h_panel, ...
            'BorderType', config.plot_panel_2d.status_panel.border_type, ...
            'BackgroundColor', config.general.window_bg_color);
        h_panels.control = uipanel('Parent', h_panel, ...
            'BorderType',config.plot_panel_2d.control_panel.border_type, ...
            'BackgroundColor', config.general.window_bg_color);
        fn = fieldnames(h_panels);
    end

    function success = fn_update_data(new_data) %this handles new data being sent to the plot panel
        if ~ishandle(h_panel)
            success = 0;
            return;
        else
            success = 1;
        end;
        orig_data = new_data;
        
        if ~isempty(orig_data)
            %global max is always the maximum value in the data
            options.global_max = max(max(abs(orig_data.f)));
            if options.auto_normalise
                options.norm_val = options.global_max;
            end
            %other limits are only set if undefined
            %             keyboard
            if isempty(options.max_val)|isempty(options.db_range)|isempty(options.norm_val)
                %initial set of limits (for plotting) and normalisation value
                options.norm_val = options.global_max; %value against which everything is normalised, including max_val
                options.max_val = 1; %maximum normalised value to plot (used for linear and dB scales)
                options.db_range = 40; %db range for dB scale only
                fn_set_sliders;
            end;
            %sort out x and z axes to be just vectors
            if size(orig_data.x, 1) > 1 & size(orig_data.x, 2) > 1
                if orig_data.x(1,1) == orig_data.x(1,2)
                    orig_data.x = orig_data.x(:,1);
                    orig_data.z = orig_data.z(1,:);
                else
                    orig_data.x = orig_data.x(1,:);
                    orig_data.z = orig_data.z(:,1);
                end;
            end;
            orig_data.x = orig_data.x(:);
            orig_data.z = orig_data.z(:);
            %set global axis limits
            options.global_x_lim = [min(orig_data.x), max(orig_data.x)];
            %setup x limits if not already set
            if isempty(options.x_lim)
                options.x_lim = options.global_x_lim;
            end;
            for ii = 1:size(orig_data.f, 1)
                set(findall(h_toolbar, 'Tag', sprintf('show_line_%i', ii)), 'Visible', 'On');
            end
            for ii = size(orig_data.f, 1)+1:max_lines
                set(findall(h_toolbar, 'Tag', sprintf('show_line_%i', ii)), 'Visible', 'Off');
            end
            
            fn_update_graphs;
            set(h_axes.main, 'ButtonDownFcn', @cb_button_down_main);
            drawnow;
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
        %this preserves the width of side control bar to absolute
        %pixel value
        p = getpixelposition(h_panel);
        setpixelposition(h_panels.control, [p(3) - slider_width_pixels, status_height_pixels + 1, slider_width_pixels, p(4) - status_height_pixels]);
        setpixelposition(h_panels.status, [1, 1, p(3), status_height_pixels]);
        setpixelposition(h_panels.plot, [1, status_height_pixels + 1, p(3) - slider_width_pixels, p(4) - status_height_pixels]);
        fn_resize_control_bar;
    end

    function fn_update_layout
        p = [0, 0, 1, 1];
        set(h_panels.main, 'Position', p);
        drawnow;
    end

    function cb_control(src, ev)
        tag = get(src, 'Tag');
        switch tag
            case 'view.zoomin'
                fn_radio_group2(h_toolbar, tag, 0);
                fn_zoom;
            case 'zoomout'
                %                 fn_radio_group2(h_toolbar, tag, 0);
                fn_zoom_out;
            case 'view.pan'
                fn_radio_group2(h_toolbar, tag, 0);
                fn_pan;
            case 'south'
                fn_update_layout;
            case 'west'
                fn_update_layout;
            case 'colorbar'
                fn_colorbar;
            case 'cursor.point'
                fn_radio_group2(h_toolbar, tag, 0);
                fn_update_graphs;
            case 'cursor.region'
                fn_radio_group2(h_toolbar, tag, 0);
                fn_update_graphs;
            case 'scale.linear'
                fn_radio_group2(h_toolbar, tag, 1);
                reset_y_lim = 1;
                fn_update_graphs;
                fn_set_sliders;
            case 'scale.log'
                fn_radio_group2(h_toolbar, tag, 1);
                reset_y_lim = 1;
                fn_update_graphs;
                fn_set_sliders;
            case 'plotwhat.real'
                fn_radio_group2(h_toolbar, tag, 1);
                reset_y_lim = 1;
                fn_update_graphs;
                fn_set_sliders;
            case 'plotwhat.mod'
                fn_radio_group2(h_toolbar, tag, 1);
                reset_y_lim = 1;
                fn_update_graphs;
                fn_set_sliders;
            case 'plotwhat.arg'
                fn_radio_group2(h_toolbar, tag, 1);
                reset_y_lim = 1;
                fn_update_graphs;
                fn_set_sliders;
            case 'custom'
                fn_handle_custom_button_push(get(src, 'String'));
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
            case 'auto_normalise'
                fn_auto_normalise;
                fn_update_graphs;
            otherwise
                fn_update_graphs;
        end
    end

    function fn_normalise
        if isempty(options.select) | (~fn_get_control_status(h_toolbar, 'cursor.point') & ~fn_get_control_status(h_toolbar, 'cursor.region'))
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
        if size(options.select, 1) == 2
            button = questdlg('Re-normalise to region value?','Normalisation','Peak', 'RMS', 'Cancel', 'Peak');
            if strcmp(button, 'Cancel')
                return;
            end
            %normalise to selected region
            switch button
                case 'RMS'
                    options.norm_val = abs(fn_get_rms_in_region(plot_data, options.select));
                case 'Peak'
                    options.norm_val = abs(fn_get_max_in_region(plot_data, options.select));
            end
            options.max_val = 1;
        end
    end

    function fn_auto_normalise
        if fn_get_control_status(h_toolbar, 'auto_normalise')
            options.max_val = 1;
            options.norm_val = options.global_max;
            options.auto_normalise = 1;
        else
            options.auto_normalise = 0;
        end
    end

    function fn_handle_custom_button_push(nm)
        for ii = 1:length(options.custom_button)
            if strcmp(options.custom_button(ii).string, nm)
                options.clicked_custom_button_index = ii;
                feval(options.custom_button(ii).function, options);
                options.clicked_custom_button_index = [];
                return;
            end;
        end
    end


    function cb_button_down_main(src, eventdata)
        tmp = get(src, 'CurrentPoint');
        tmp = tmp(1,1:2);
        if fn_get_control_status(h_toolbar, 'cursor.point')
            options.select = tmp ./ [options.x_axis_sf, options.z_axis_sf];
            fn_update_cursor;
            fn_update_graphs;
        end;
        if fn_get_control_status(h_toolbar, 'cursor.region')
            if next_point == 1
                options.select = tmp ./ [options.x_axis_sf, options.z_axis_sf];
                next_point = 2;
            else
                options.select = [options.select; tmp ./ [options.x_axis_sf, options.z_axis_sf]];
                next_point = 1;
            end;
            fn_update_cursor;
            fn_update_graphs;
        end;
    end

    function fn_update_graphs
        fn_set_options_to_match_toolbar_btns;
        %         fn_set_sliders;
        if isempty(orig_data)% | ~data_has_been_plotted
            %             h_plot_objects.selection_text = uicontrol(h_panels.main, 'Style', 'Text', 'String', '', 'HorizontalAlignment', 'Center', 'Units', 'Normalized', 'Position', [0, 0.9, 1, 0.1], 'BackgroundColor', panel_colour);
            %should clear graphs here - e.g. plot crosses
            return;
        end
        
        plot_data = orig_data;

        if ~data_has_been_plotted
            reset_x_lim = 1;
            reset_y_lim = 1;
        end
        
        %update main graph
        set(h_axes.main, 'Visible', 'On');
        set(get(h_axes.main, 'XLabel'), 'String', options.x_label);
        set(get(h_axes.main, 'YLabel'), 'String', options.z_label);
        [plot_val, limits, options.scale_mode] = fn_convert_to_plot_val(plot_data.f, options.plotwhat, options.scale_mode, options.db_range, options.max_val, options.norm_val);
        hold on;
        delete(findobj(h_axes.main, 'Tag', 'Ascan'));
        jj = 1;
        h = [];
        for ii = 1:min(size(plot_val, 1), max_lines)
            if fn_get_control_status(h_toolbar, sprintf('show_line_%i', ii))
                h(jj) = plot(h_axes.main, plot_data.x * options.x_axis_sf, plot_val(ii,:), line_colors(ii), 'Tag', 'Ascan', 'HitTest', 'off');
                str{jj} = sprintf('%i', ii);
                jj = jj + 1;
            end
        end
        if ~isempty(h)
            legend(h, str);
        end
        if reset_x_lim
            xlim(h_axes.main, options.x_lim * options.x_axis_sf);
            reset_x_lim = 0;
        end
        if reset_y_lim
            ylim(h_axes.main, limits);
            reset_y_lim = 0;
        end
        
              
        fn_update_cursor;
        set(h_axes.main, 'ButtonDownFcn', @cb_button_down_main);
        %plot geometric features, such as array
        delete(findall(h_axes.main,'Tag','geom'))
        if isfield(orig_data, 'geom')
            if isfield(orig_data.geom, 'array')
                line(orig_data.geom.array.x * options.x_axis_sf, orig_data.geom.array.z * options.z_axis_sf, ...
                    'Color', config.array_el_edge_color, ...
                    'Parent', h_axes.main, ...
                    'Tag', 'geom');
            end
            if isfield(orig_data.geom, 'lines')
                for ii = 1:length(orig_data.geom.lines)
                    line(orig_data.geom.lines(ii).x * options.x_axis_sf, orig_data.geom.lines(ii).z * options.z_axis_sf, ...
                        'LineStyle', orig_data.geom.lines(ii).style, ...
                        'Color', 'w', ...
                        'Parent', h_axes.main, ...
                        'Tag', 'geom');
                end
            end
        end
        
        data_has_been_plotted = 1;
        
        %Update string
        set(h_plot_objects.selection_text, 'String', pointer_str);
        set(h_plot_objects.range_text, 'String', range_str);
        
        %         fn_set_sliders;
    end


    function fn_zoom_out
        options.x_lim = options.global_x_lim;
        data_has_been_plotted = 0;
        fn_update_graphs;
    end


    function fn_zoom
        if fn_get_control_status(h_toolbar, 'view.zoomin')
            set(h_zoom, 'Enable','On');
            set(h_zoom, 'Direction','In');
        else
            set(h_zoom, 'Enable', 'Off');
        end
    end;
    
    function fn_pan
        if fn_get_control_status(h_toolbar, 'view.pan')
            pan on;
        else
            pan off;
        end;
    end;
    
    function cb_range_change(src, event_data)
        tmp = 1 - get(h_toolbar_btns.range_slider, 'value');
        if strcmpi(options.scale_mode, 'dB')
            %slider here is interpreted as dB dynamic range from 1 to 120
            db = -(tmp * (config.plot_panel_2d.max_db_range - 1) + 1);
            options.min_val = options.max_val * 10 ^ (db / 20);
        end;
        if strcmpi(options.scale_mode, 'linear')
            %slider here is interpreted as max of linear range
            options.min_val = options.global_max * 10 ^ (-config.plot_panel_2d.max_db_range / 20);
            options.max_val = tmp * (options.global_max - options.min_val) + options.min_val;
            options.min_val = 0;
        end;
        [dummy, limits, options.scale_mode] = fn_convert_to_plot_val(0, options.plotwhat, options.scale_mode, options.db_range, options.max_val, options.norm_val)
        fn_update_graphs;
    end;
    
    function fn_update_cursor
        c = axis(h_axes.main);
        if fn_get_control_status(h_toolbar, 'cursor.point')
            set(h_plot_objects.region, 'visible', 'off');
            if size(options.select, 1)
                set(h_plot_objects.z_crosshair, 'XData', ones(1,2) * options.select(1,1) * options.x_axis_sf, 'YData', [c(3), c(4)] , 'Visible', 'On');
                pos_str = sprintf('%.2f', options.select(1,1) * options.x_axis_sf);
                val_str = '';
                for line_no = 1:min(size(plot_data.f, 1), max_lines)
                    if fn_get_control_status(h_toolbar, sprintf('show_line_%i', line_no))
                        switch options.scale_mode
                            case 'linear'
                                val_str = [val_str, sprintf('(%i) %.3f %% ', line_no, abs(fn_get_value_at_point(plot_data, options.select, line_no)) / options.norm_val * 100)];
                            case 'log'
                                val_str = [val_str, sprintf('(%i) %.1f dB ', line_no, 20*log10(abs(fn_get_value_at_point(plot_data, options.select, line_no)) / options.norm_val))];
                        end
                    end
                end
                pointer_str = {pos_str, val_str};
            end
        end
        if fn_get_control_status(h_toolbar, 'cursor.region')
            set(h_plot_objects.z_crosshair, 'visible', 'off');
            if size(options.select, 1) == 0
                set(h_plot_objects.region, 'visible', 'off');
            else
                if size(options.select, 1) == 1
                    set(h_plot_objects.region, 'XData', ones(1,2) * options.select(1,1) * options.x_axis_sf, 'YData', [c(3), c(4)] , 'Visible', 'On');
                    pointer_str = sprintf('%.2f to ...', options.select(1,1) * options.x_axis_sf);
                else
                    x = [options.select(1,1), options.select(2,1), options.select(2,1), options.select(1,1), options.select(1,1)] * options.x_axis_sf;
                    y = [c(3), c(3), c(4), c(4), c(3)];
                    set(h_plot_objects.region, 'XData', x, 'YData', y, 'visible', 'on');
                    pos_str = sprintf('%.2f to %.2f', [options.select(1,1)' * options.x_axis_sf, options.select(2,1)' * options.x_axis_sf]);
                    val_str = 'Maxima ';
                    for line_no = 1:min(size(plot_data.f, 1), max_lines)
                        if fn_get_control_status(h_toolbar, sprintf('show_line_%i', line_no))
                            switch options.scale_mode
                                case 'linear'
                                    val_str = [val_str, sprintf('(%i) %.3f %% ', line_no, abs(fn_get_max_in_region(plot_data, options.select, line_no)) / options.norm_val * 100)];
                                case 'log'
                                    val_str = [val_str, sprintf('(%i) %.1f dB ', line_no, 20*log10(abs(fn_get_max_in_region(plot_data, options.select, line_no)) / options.norm_val))];
                            end
                        end
                    end
                    pointer_str = {pos_str, val_str};
                end;
            end;
        end;
        if ~fn_get_control_status(h_toolbar, 'cursor.point') & ~fn_get_control_status(h_toolbar, 'cursor.region')
            set(h_plot_objects.z_crosshair, 'visible', 'off');
            set(h_plot_objects.region, 'visible', 'off');
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

function val = fn_get_value_at_point(data, pt, line_no)
i1 = interp1(data.x, [1:length(data.x)], pt(1), 'nearest');
if i1 >= 1 & i1 <= size(data.f, 2) & line_no >= 1 & line_no <= size(data.f, 1)
    val = data.f(line_no, i1);
else
    val = 0;
end
end

function val = fn_get_rms_in_region(data, reg, line_no)
i1 = find(data.x >= min(reg(:, 1)) & data.x <= max(reg(:, 1)));
if min(i1) >= 1 & max(i1) <= size(data.f, 2) & line_no >= 1 & line_no <= size(data.f, 1)
    val = data.f(line_no, i1);
else
    val = 0;
end
val = sqrt(mean(val(:) .* conj(val(:))));
end

function val = fn_get_max_in_region(data, reg, line_no)
i1 = find(data.x >= min(reg(:, 1)) & data.x <= max(reg(:, 1)));
if min(i1) >= 1 & max(i1) <= size(data.f, 2) & line_no >= 1 & line_no <= size(data.f, 1)
    val = data.f(line_no, i1);
else
    val = 0;
end
val = max(abs(val(:)));
end

function res = fn_set_visible(h, val)
if val
    set(h, 'Visible', 'On');
else
    set(h, 'Visible', 'Off');
end;
end