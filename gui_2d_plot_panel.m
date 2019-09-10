function [h_fn_update_data, h_fn_get_options, h_fn_set_options] = gui_2d_plot_panel(h_panel, h_toolbar)
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
default_options.x_axis_sf = 1000;
default_options.z_axis_sf = 1000;
default_options.cursor_type = 'none';
default_options.axis_equal = 1;
default_options.show_south_axis = 0;
default_options.show_west_axis = 0;
default_options.show_colorbar = 0;
default_options.interpolate = 0;
default_options.monochrome = 0;
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

orig_data = [];
plot_data = [];

%load icons
icons = [];
load(config.files.icon_file);

%define all the objects
h_panels = fn_create_panels;

%add the axes to the graph panels and link appropriate ones
[h_axes, h_link] = fn_create_axes;

%add the actual plot objects (these always exist, they are just updated in
%position/visibility in operation)
h_plot_objects = fn_create_plot_objects;

%various handles
h_toolbar_btns = [];
h_max_slider = [];
h_range_slider = [];
h_custom_button = [];

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
        if options.show_west_axis
            fn_set_control_status(h_toolbar, 'west', 1);
        else
            fn_set_control_status(h_toolbar, 'west', 0);
        end
        if options.show_south_axis
            fn_set_control_status(h_toolbar, 'south', 1);
        else
            fn_set_control_status(h_toolbar, 'south', 0);
        end
        if options.show_colorbar
            fn_set_control_status(h_toolbar, 'colorbar', 1);
        else
            fn_set_control_status(h_toolbar, 'colorbar', 0);
        end
        if options.monochrome
            fn_set_control_status(h_toolbar, 'monochrome', 1);
        else
            fn_set_control_status(h_toolbar, 'monochrome', 0);
        end
        if options.axis_equal
            fn_set_control_status(h_toolbar, 'axis_equal', 1);
        else
            fn_set_control_status(h_toolbar, 'axis_equal', 0);
        end
        if options.interpolate
            fn_set_control_status(h_toolbar, 'interpolate', 1);
        else
            fn_set_control_status(h_toolbar, 'interpolate', 0);
        end
        if options.auto_normalise
            fn_set_control_status(h_toolbar, 'auto_normalise', 1);
        else
            fn_set_control_status(h_toolbar, 'auto_normalise', 0);
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
        h_toolbar_btns(4).tag = 'south';
        h_toolbar_btns(4).tooltip = 'Show bottom axis';
        h_toolbar_btns(4).icon = fn_get_cdata_for_named_icon(icons, 'south');
        h_toolbar_btns(4).type = 'toggle';
        
        h_toolbar_btns(5).tag = 'west';
        h_toolbar_btns(5).tooltip = 'Show side axis';
        h_toolbar_btns(5).icon = fn_get_cdata_for_named_icon(icons, 'west');
        h_toolbar_btns(5).type = 'toggle';
        
        h_toolbar_btns(6).tag = 'colorbar';
        h_toolbar_btns(6).tooltip = 'Show colorbar';
        h_toolbar_btns(6).icon = fn_get_cdata_for_named_icon(icons, 'Annotation.InsertColorbar');
        h_toolbar_btns(6).type = 'toggle';
        
        h_toolbar_btns(7).separator = 1;
        h_toolbar_btns(7).tag = 'cursor.point';
        h_toolbar_btns(7).tooltip = 'Select point';
        h_toolbar_btns(7).icon = fn_get_cdata_for_named_icon(icons, 'point');
        h_toolbar_btns(7).type = 'toggle';
        
        h_toolbar_btns(8).tag = 'cursor.region';
        h_toolbar_btns(8).tooltip = 'Select region';
        h_toolbar_btns(8).icon = fn_get_cdata_for_named_icon(icons, 'region');
        h_toolbar_btns(8).type = 'toggle';
        
        h_toolbar_btns(9).separator = 1;
        h_toolbar_btns(9).tag = 'scale.linear';
        h_toolbar_btns(9).tooltip = 'Linear scale';
        h_toolbar_btns(9).icon = fn_get_cdata_for_named_icon(icons, 'linear');
        h_toolbar_btns(9).type = 'toggle';
        
        h_toolbar_btns(10).tag = 'scale.log';
        h_toolbar_btns(10).tooltip = 'Log scale';
        h_toolbar_btns(10).icon = fn_get_cdata_for_named_icon(icons, 'log');
        h_toolbar_btns(10).type = 'toggle';
        
        h_toolbar_btns(11).tag = 'plotwhat.real';
        h_toolbar_btns(11).tooltip = 'Plot real part';
        h_toolbar_btns(11).icon = fn_get_cdata_for_named_icon(icons, 'real');
        h_toolbar_btns(11).type = 'toggle';
        h_toolbar_btns(11).separator = 1;
        
        h_toolbar_btns(12).tag = 'plotwhat.mod';
        h_toolbar_btns(12).tooltip = 'Plot modulus';
        h_toolbar_btns(12).icon = fn_get_cdata_for_named_icon(icons, 'modulus');
        h_toolbar_btns(12).type = 'toggle';
        
        h_toolbar_btns(13).tag = 'plotwhat.arg';
        h_toolbar_btns(13).tooltip = 'Plot argument';
        h_toolbar_btns(13).icon = fn_get_cdata_for_named_icon(icons, 'argument');
        h_toolbar_btns(13).type = 'toggle';
        
        h_toolbar_btns(14).tag = 'normalise';
        h_toolbar_btns(14).tooltip = 'Normalise';
        h_toolbar_btns(14).icon = fn_get_cdata_for_named_icon(icons, 'normalise');
        h_toolbar_btns(14).type = 'pushtool';
        
        h_toolbar_btns(15).tag = 'auto_normalise';
        h_toolbar_btns(15).tooltip = 'Auto normalise';
        h_toolbar_btns(15).icon = fn_get_cdata_for_named_icon(icons, 'auto_normalise');
        h_toolbar_btns(15).type = 'toggle';
        
        h_toolbar_btns(16).separator = 1;
        h_toolbar_btns(16).tag = 'axis_equal';
        h_toolbar_btns(16).tooltip = 'Set aspect ratio to one';
        h_toolbar_btns(16).icon = fn_get_cdata_for_named_icon(icons, 'axisequal');
        h_toolbar_btns(16).type = 'toggle';
        
        h_toolbar_btns(17).tag = 'interpolate';
        h_toolbar_btns(17).tooltip = 'Interpolate image';
        h_toolbar_btns(17).icon = fn_get_cdata_for_named_icon(icons, 'interpolate');
        h_toolbar_btns(17).type = 'toggle';
        
        h_toolbar_btns(18).tag = 'monochrome';
        h_toolbar_btns(18).tooltip = 'Monochrome image';
        h_toolbar_btns(18).icon = fn_get_cdata_for_named_icon(icons, 'monochrome');
        h_toolbar_btns(18).type = 'toggle';
        
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
        %add custom buttons up the bottom of plot display (above status bar)
        ch = findall(h_panels.control2, 'tag', 'custom');
        if ~isempty(ch)
            delete(ch);
        end
        radio_group=[]; radio_group2=[];
        %set the new controls
        if ~isempty(options.custom_button)
            for ii = 1:length(options.custom_button)
                if (isfield(options.custom_button(ii),'style'))
                    if (strcmp(options.custom_button(ii).style,'radiobutton')>0)
                        if (length(radio_group) < options.custom_button(ii).group || isempty(radio_group(options.custom_button(ii).group)))
                            radio_group2(options.custom_button(ii).group)=ii;
                            k=0;
                            for jj=ii:length(options.custom_button)
                                if (options.custom_button(jj).group == options.custom_button(ii).group)
                                    k=k+1;
                                end
                            end

                            %k=sum(options.custom_button(:).group == options.custom_button(ii).group);
                            radio_group(options.custom_button(ii).group) = uibuttongroup(h_panels.control2,'Units', 'pixels','Position',[10+70*(ii-1) 0 70*k+10 20]) %,'SelectionChangedFcn',@cb_control);
                            
                        end
                        
                        h_custom_button(ii).handle = uicontrol(radio_group(options.custom_button(ii).group), ...
                            'String', options.custom_button(ii).string, ...
                            'Style', options.custom_button(ii).style, ...
                            'Tag', 'custom',...
                            'Callback', @cb_control, ...
                            'HandleVisibility','off', ...
                            'Position',[10+65*(ii-radio_group2(options.custom_button(ii).group)) 0 60 20]);
                        if (ii == radio_group2(options.custom_button(ii).group))
                            fn_handle_custom_button_push(options.custom_button(ii).string,h_custom_button(ii).handle.Value);
                        end
                    else
                        h_custom_button(ii).handle = uicontrol(h_panels.control2, ...
                            'String', options.custom_button(ii).string, ...
                            'Style', options.custom_button(ii).style, ...
                            'Callback', @cb_control, ...
                            'Tag', 'custom',...
                            'Position',[10+70*(ii-1) 0 65 20]);
                        if ((strcmp(options.custom_button(ii).style,'checkbox')>0 || strcmp(options.custom_button(ii).style,'togglebutton')>0) && isfield(options.custom_button(ii),'defaultValue'))
                            h_custom_button(ii).handle.Value=options.custom_button(ii).defaultValue;
                        end    
                        fn_handle_custom_button_push(options.custom_button(ii).string,h_custom_button(ii).handle.Value);
                    end
                    
                else
                    h_custom_button(ii).handle = uicontrol(h_panels.control2, ...
                        'String', options.custom_button(ii).string, ...
                        'Style', 'pushbutton', ...
                        'Callback', @cb_control, ...
                        'Tag', 'custom');
                    fn_handle_custom_button_push(options.custom_button(ii).string,h_custom_button(ii).handle.Value);    
                end
                
            end;
            fn_resize; % force resize
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
        h_plot_objects.image = image('Parent', h_axes.main, 'Visible', 'Off', 'CData', eye(2), 'CDataMapping', 'Scaled', 'HitTest', 'off');
        h_plot_objects.x_crosshair = plot(h_axes.main, [0,0], [1,1], 'Visible', 'Off', 'LineStyle', ':', 'Color', [1,1,1], 'HitTest', 'off');
        h_plot_objects.z_crosshair = plot(h_axes.main, [0,0], [1,1], 'Visible', 'Off', 'LineStyle', ':', 'Color', [1,1,1], 'HitTest', 'off');
        h_plot_objects.region = plot(h_axes.main, [0,0], [1,1], 'Visible', 'Off', 'LineStyle', ':', 'Marker', 's', 'Color', [1,1,0], 'HitTest', 'off');
        
        h_plot_objects.selection_text = uicontrol(h_panels.status, 'Style', 'Text', 'String', {'1','2','3'}, 'HorizontalAlignment', 'Left', 'Units', 'Normalized', 'Position', [0.02, 0, 0.5, 1], 'BackgroundColor', panel_colour);
        h_plot_objects.range_text = uicontrol(h_panels.status, 'Style', 'Text', 'String', {'1','2','3'}, 'HorizontalAlignment', 'Right', 'Units', 'Normalized', 'Position', [0.5, 0, 0.48, 1], 'BackgroundColor', panel_colour);
    end

    function [h_axes, h_link] = fn_create_axes
        h_axes.main = axes('Parent', h_panels.main, 'NextPlot', 'Add', 'Layer', 'Top');
        if options.axis_equal
            set(h_axes.main, 'DataAspectRatio', [1,1,1]);
        end;
        h_axes.south = axes('Parent', h_panels.south, 'XTick', [], 'NextPlot', 'replacechildren');
        h_axes.west = axes('Parent', h_panels.west, 'YTick', [], 'NextPlot', 'replacechildren', 'YDir', 'Reverse');
        h_link.x = linkprop([h_axes.main, h_axes.south], 'XLim');
        h_link.y = linkprop([h_axes.main, h_axes.west], 'YLim');
    end

    function h_panels = fn_create_panels
        h_panels.plot = uipanel('Parent', h_panel);
        h_panels.main = uipanel('Parent', h_panels.plot, ...
            'BorderType',  config.plot_panel_2d.graph_panel.border_type, ...
            'BackgroundColor', config.general.window_bg_color);
        h_panels.south = uipanel('Parent', h_panels.plot, ...
            'BorderType',  config.plot_panel_2d.graph_panel.border_type, ...
            'BackgroundColor', config.general.window_bg_color);
        h_panels.west = uipanel('Parent', h_panels.plot, ...
            'BorderType',  config.plot_panel_2d.graph_panel.border_type, ...
            'BackgroundColor', config.general.window_bg_color);
        h_panels.status = uipanel('Parent', h_panel, ...
            'BorderType', config.plot_panel_2d.status_panel.border_type, ...
            'BackgroundColor', config.general.window_bg_color);
        h_panels.control = uipanel('Parent', h_panel, ...
            'BorderType',config.plot_panel_2d.control_panel.border_type, ...
            'BackgroundColor', config.general.window_bg_color);
        h_panels.control2 = uipanel('Parent', h_panel, ...
            'BorderType', config.plot_panel_2d.status_panel.border_type, ...
            'BackgroundColor', config.general.window_bg_color);  
        fn = fieldnames(h_panels);
        %         for ii = 1:length(fn)
        %             set(getfield(h_panels, fn{ii}), ...
        %                 'BorderType',  config.plot_panel_2d.graph_panel.border_type, ...
        %                 'BackgroundColor', config.general.window_bg_color);
        %         end
        %         set(h_panels.status, ...
        %             'BorderType', config.plot_panel_2d.status_panel.border_type, ...
        %             'BackgroundColor', config.general.window_bg_color);
        %         set(h_panels.control, ...
        %             'BorderType',config.plot_panel_2d.control_panel.border_type, ...
        %             'BackgroundColor', config.general.window_bg_color);
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
                if (isfield(options,'levelling_active') && options.levelling_active>0 && isfield(orig_data,'mask'))
                    options.norm_val = 1;
                else
                    if isfield(orig_data,'normalisation_factor')
                        options.norm_val = 1; % Data has been preivously normalised, so use 1
                    else
                        options.norm_val = options.global_max;
                    end
                end
            end
            %other limits are only set if undefined
            %             keyboard
            if isempty(options.max_val)|isempty(options.db_range)|isempty(options.norm_val)
                %initial set of limits (for plotting) and normalisation value
                if (isfield(options,'levelling_active') && options.levelling_active>0 && isfield(orig_data,'mask'))
                    options.norm_val = 1;
                else
                    if isfield(orig_data,'normalisation_factor')
                        options.norm_val = 1; % Data has been preivously normalised, so use 1
                    else
                        options.norm_val = options.global_max; %value against which everything is normalised, including max_val
                    end
                end
                options.max_val = 1; %maximum normalised value to plot (used for linear and dB scales)
                if (isfield(orig_data,'combined_plot') || isfield(orig_data,'mask'))
                    if (isfield(options,'levelling_active') && options.levelling_active>0 && isfield(orig_data,'mask'))
                        options.db_range = 24; %db range for dB scale only
                    else
                        options.db_range = 60; %db range for dB scale only
                    end
                else
                    options.db_range = 40; %db range for dB scale only
                end
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
            options.global_z_lim = [min(orig_data.z), max(orig_data.z)];
            %setup x and z limits if not already set
            if isempty(options.x_lim)|isempty(options.z_lim)
                options.x_lim = options.global_x_lim;
                options.z_lim = options.global_z_lim;
            end;
            
            fn_update_graphs;
            set(h_axes.main, 'ButtonDownFcn', @cb_button_down_main);
            drawnow;
        else
            %no data - turn off display
            %             set(h_axes.main, 'Visible', 'Off');
            options.global_max = 1;
            options.norm_val = 1;
            options.max_val = 1;
            if (isfield(orig_data,'combined_plot') || isfield(orig_data,'mask'))
                if (isfield(options,'levelling_active') && options.levelling_active>0 && isfield(orig_data,'mask'))
                    options.db_range = 24; %db range for dB scale only
                else
                    options.db_range = 60; %db range for dB scale only
                end
            else
                options.db_range = 40; %db range for dB scale only
            end
        end
    end

    function fn_resize(src, evt)
        %this preserves the width of side control bar to absolute
        %pixel value
        p = getpixelposition(h_panel);
        %         p(1:2) = p(1:2) + 1;
        %         p(3:4) = p(3:4) - 2;
        setpixelposition(h_panels.control, [p(3) - slider_width_pixels, status_height_pixels + 1, slider_width_pixels, p(4) - status_height_pixels]);
        setpixelposition(h_panels.status, [1, 1, p(3), status_height_pixels]);
        if (isfield(options,'custom_button') && ~isempty(options.custom_button))
            control2_height_pixels=round(status_height_pixels*0.5);
        else
            control2_height_pixels=0;
        end
        setpixelposition(h_panels.control2, [1, status_height_pixels + 1, p(3)- slider_width_pixels, control2_height_pixels]);
        setpixelposition(h_panels.plot, [1, status_height_pixels + control2_height_pixels + 1, p(3) - slider_width_pixels, p(4) - status_height_pixels - control2_height_pixels]);
        fn_resize_control_bar;
    end

    function fn_update_layout
        %this just does the layout, not the content and not the actual
        %plotting
        p = [0, 0, 1, 1];
        if fn_get_control_status(h_toolbar, 'south')
            options.show_south_axis = 1;
            p(2) = config.plot_panel_2d.side_graph_fraction;
            p(4) = 1 - config.plot_panel_2d.side_graph_fraction;
        else
            options.show_south_axis = 0;
        end;
        if fn_get_control_status(h_toolbar, 'west')
            options.show_west_axis = 1;
            p(1) = config.plot_panel_2d.side_graph_fraction;
            p(3) = 1 - config.plot_panel_2d.side_graph_fraction;
        else
            options.show_west_axis = 0;
        end;
        set(h_panels.main, 'Position', p);
        set(h_panels.south, 'Position', [p(1), 0, p(3), config.plot_panel_2d.side_graph_fraction]);
        set(h_panels.west, 'Position', [0, p(2), config.plot_panel_2d.side_graph_fraction, p(4)]);
        fn_set_visible(h_panels.south, fn_get_control_status(h_toolbar, 'south'));
        fn_set_visible(h_panels.west, fn_get_control_status(h_toolbar, 'west'));
        drawnow;
    end

    function cb_control(src, ev)
        tag = get(src, 'Tag');
        switch tag
            case 'view.zoomin'
%                 fn_radio_group2(h_toolbar, tag, 0);
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
                fn_update_graphs;
                fn_set_sliders;
            case 'scale.log'
                fn_radio_group2(h_toolbar, tag, 1);
                fn_update_graphs;
                fn_set_sliders;
            case 'plotwhat.real'
                fn_radio_group2(h_toolbar, tag, 1);
                fn_update_graphs;
                fn_set_sliders;
            case 'plotwhat.mod'
                fn_radio_group2(h_toolbar, tag, 1);
                fn_update_graphs;
                fn_set_sliders;
            case 'plotwhat.arg'
                fn_radio_group2(h_toolbar, tag, 1);
                fn_update_graphs;
                fn_set_sliders;
            case 'custom'
                fn_handle_custom_button_push(get(src, 'String'),src.Value);
                fn_update_graphs
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
            case 'axis_equal'
                fn_axis_equal;
                fn_update_graphs;
            case 'interpolate'
                fn_interpolate;
                fn_update_graphs;
            case 'monochrome';
                fn_monochrome;
                fn_update_graphs;
        end
    end

    function fn_monochrome
        if fn_get_control_status(h_toolbar, 'monochrome')
            options.monochrome = 1;
        else
            options.monochrome = 0;
        end
    end

    function fn_axis_equal
        if fn_get_control_status(h_toolbar, 'axis_equal')
            options.axis_equal = 1;
        else
            options.axis_equal = 0;
        end
    end

    function fn_interpolate
        if fn_get_control_status(h_toolbar, 'interpolate')
            options.interpolate = 1;
        else
            options.interpolate = 0;
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
            options.norm_val = abs(fn_get_value_at_point(plot_data, options.select))
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

    function fn_handle_custom_button_push(nm,value)
        for ii = 1:length(options.custom_button)
            if strcmp(options.custom_button(ii).string, nm)
                options.clicked_custom_button_index = ii;
                options.value=value;
                options=feval(options.custom_button(ii).function, options);
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
                options.select
                next_point = 2;
            else
                options.select = [options.select; tmp ./ [options.x_axis_sf, options.z_axis_sf]];
                options.select
                next_point = 1;
            end;
            fn_update_cursor;
            fn_update_graphs;
        end;
    end

    function fn_update_graphs
        fn_set_options_to_match_toolbar_btns;
        %         fn_set_sliders;
        if options.axis_equal
            set(h_axes.main, 'DataAspectRatioMode', 'Manual');
        else
            set(h_axes.main, 'DataAspectRatioMode', 'Auto');
        end;
        if isempty(orig_data)% | ~data_has_been_plotted
            %             h_plot_objects.selection_text = uicontrol(h_panels.main, 'Style', 'Text', 'String', '', 'HorizontalAlignment', 'Center', 'Units', 'Normalized', 'Position', [0, 0.9, 1, 0.1], 'BackgroundColor', panel_colour);
            %should clear graphs here - e.g. plot crosses
            return;
        end
        
        if ~isempty(options.custom_button)
            for ii = 1:length(options.custom_button)
                if (isfield(options.custom_button(ii),'enable_function') && ~isempty(options.custom_button(ii).enable_function))
                    status=feval(options.custom_button(ii).enable_function,orig_data,options);
                    h_custom_button(ii).handle.Enable=status;
                end
            end
        end
        
        %mask if necessary
        if (isfield(orig_data,'mask') && isfield(orig_data,'masking_function'))
            orig_data=feval(orig_data.masking_function,orig_data,options);
            
            
            if options.auto_normalise
                options.global_max = orig_data.fmax;
                if (isfield(options,'levelling_active') && options.levelling_active>0 && isfield(orig_data,'mask'))
                    options.norm_val = 1;
                else
                    if isfield(orig_data,'normalisation_factor')
                        options.norm_val = 1; % Data has been preivously normalised, so use 1
                    else
                        options.norm_val = options.global_max;
                    end
                end
            end
            
            if (isfield(options,'pvalues_active') && options.pvalues_active > 0)
                options.scale_mode='linear';
            else
                options.scale_mode='log';
            end
            switch options.scale_mode
                case 'log'
                    fn_set_control_status(h_toolbar, 'scale.log', 1);
                    fn_set_control_status(h_toolbar, 'scale.linear', 0);
                case 'linear'
                    fn_set_control_status(h_toolbar, 'scale.log', 0);
                    fn_set_control_status(h_toolbar, 'scale.linear', 1);
            end
            fn_set_sliders;
        end
        
        %Adjust max value if levelling button has been activated
        if (((isfield(orig_data,'display_type') && ~isfield(options,'display_type_old')) || (isfield(orig_data,'display_type') && strcmp(orig_data.display_type,options.display_type_old)<1)) )
            options.levelling_changed=0;
            options.display_type_old=orig_data.display_type;
                
            %adjust range value for levelling & combined plots, otherwise default to 40
            if (isfield(orig_data,'combined_plot') || isfield(orig_data,'mask'))
                if (isfield(options,'levelling_active') && options.levelling_active>0 && isfield(orig_data,'mask'))
                    options.db_range = 24; %db range for dB scale only
                else
                    options.db_range = 60; %db range for dB scale only
                end
            else
                options.db_range = 40; %db range for dB scale only
            end
            if (strcmp(options.scale_mode,'log')>0)
                if (isfield(options,'levelling_active') && options.levelling_active>0 && isfield(orig_data,'mask'))
                   
                    h_max_slider.Value=12+50;
                    fn_range_change(h_max_slider);
                else
                    h_max_slider.Value=0+50;
                    fn_range_change(h_max_slider);
                end
            end
            
            if isfield(options,'pvalues_active') && options.pvalues_active > 0
                options.global_max=1.0;
                options.norm_val=1.0;
                options.max_val=0.01;               
            
            end
        end
        
        %interpolate
        if isfield(h_fn_get_options(),'gpu')
            if options.interpolate
                p = getpixelposition(h_axes.main);
                nx2 = single(round(p(3)));
                nz2 = single(round(p(4)));
                nx2_gpu = gpuArray(nx2);
                nz2_gpu = gpuArray(nz2);
                [nz1, nx1] = size(orig_data.f);
                plot_data.f = interpft_gpu(interpft_gpu(orig_data.f, nx2_gpu, 2), nz2_gpu, 1);
                plot_data.f=gather(plot_data.f);
                dx1 = orig_data.x(2) - orig_data.x(1);
                dz1 = orig_data.z(2) - orig_data.z(1);
                plot_data.x = orig_data.x(1) + dx1 * [0: nx2 - 1] / nx2 * nx1;
                plot_data.z = orig_data.z(1) + dz1 * [0: nz2 - 1] / nz2 * nz1;
            else
                plot_data = orig_data;
                plot_data.f=gather(plot_data.f);
            end
            
            options.max_val=gather(options.max_val);
            options.norm_val=gather(options.norm_val);
        else
            if options.interpolate
                p = getpixelposition(h_axes.main);
                nx2 = round(p(3));
                nz2 = round(p(4));
                dx1 = orig_data.x(2) - orig_data.x(1);
                dz1 = orig_data.z(2) - orig_data.z(1);
                [nz1, nx1] = size(orig_data.f);
                plot_data.f = interpft(interpft(orig_data.f, nx2, 2), nz2, 1);
                plot_data.x = orig_data.x(1) + dx1 * [0: nx2 - 1] / nx2 * nx1;
                plot_data.z = orig_data.z(1) + dz1 * [0: nz2 - 1] / nz2 * nz1;
            else
                plot_data = orig_data;
                
            end
        end
        
        if (isfield(options,'pvalues_active') && options.pvalues_active>1 && isfield(orig_data,'f2'))
            plot_data.f=plot_data.f2;
        end
        
        %update main graph
        set(h_axes.main, 'Visible', 'On');
        set(get(h_axes.main, 'XLabel'), 'String', options.x_label);
        set(get(h_axes.main, 'YLabel'), 'String', options.z_label);
        [plot_val, limits, options.scale_mode] = fn_convert_to_plot_val(plot_data.f, options.plotwhat, options.scale_mode, options.db_range, options.max_val, options.norm_val);
        set(h_plot_objects.image, ...
            'XData', [min(plot_data.x), max(plot_data.x)] * options.x_axis_sf, ...
            'YData', [min(plot_data.z), max(plot_data.z)] * options.z_axis_sf, ...
            'CData' , plot_val, ...
            'Visible', 'On');
            
        %Set NaN values to transparent 
        set(h_plot_objects.image,'alphadata',~isnan(plot_val));         
            
        if ~data_has_been_plotted
            axis(h_axes.main, [options.global_x_lim * options.x_axis_sf, options.global_z_lim * options.z_axis_sf]);
        end
        
        %Axis scale if combined
        if (isfield(orig_data,'combined_plot') && orig_data.combined_plot>0 && (isfield(options,'pvalues_active') && options.pvalues_active<2  || ~isfield(options,'pvalues_active')))
            full_width=orig_data.xfull;
            full_height=orig_data.zfull;
            dx=orig_data.x(2)-orig_data.x(1);
            dz=orig_data.z(2)-orig_data.z(1);
            dz=dz*full_height;
            dx=dx*full_width;
            dz=dz/dx;
            vec1=[dz 1 1];
            daspect(vec1);
            set(h_axes.main, 'DataAspectRatio', vec1);
            %disp(['daspect ',num2str(dz),' 1 1']);
        end
        
        try
            caxis(h_axes.main, limits);
        catch
            % Catch if limits are invalid
            %disp('Invalid limit. Ignoring')
        end
        
        set(h_axes.main, 'XTickMode','auto');
        set(h_axes.main, 'YTickMode','auto');
        set(h_axes.main, 'XTickLabelMode','auto');
        set(h_axes.main, 'YTickLabelMode','auto');
        delete(findall(h_axes.main,'Tag','text1'))
        if (isfield(orig_data,'combined_plot') && orig_data.combined_plot>0 && (isfield(options,'pvalues_active') && options.pvalues_active<2  || ~isfield(options,'pvalues_active'))) %% Added combined plot tick control RB 2018/12/17
            cur_axis=get(h_axes.main);
            
            %convert labels for multi-view setup
            ii=1; start_subimage=orig_data.xspacing(1,ii); end_subimage=orig_data.xspacing(2,1); 
            for i=1:length(cur_axis.XTick)
                cur_pos=(cur_axis.XTick(i)/options.x_axis_sf-options.global_x_lim(1))/(options.global_x_lim(2)-options.global_x_lim(1));
                while (cur_pos > end_subimage)
                    ii=ii+1;
                    start_subimage=orig_data.xspacing(1,ii);
                    end_subimage=orig_data.xspacing(2,ii);
                end
                if (cur_pos < start_subimage)
                    cur_val(i)=NaN;
                else
                    cur_val(i)=round(10*(((cur_pos - start_subimage) / (end_subimage - start_subimage) * (options.global_x_lim(2)-options.global_x_lim(1))+options.global_x_lim(1)) * options.x_axis_sf))/10.0;
                end
            end
            iNaN=find(isnan(cur_val));
            XTickLabel1=num2cell(cur_val.');
            if (length(iNaN)>0)
                try
                    XTickLabel1{iNaN}='';
                catch
                
                end
            end
            set(h_axes.main, 'XTickLabel',XTickLabel1);
            set(h_axes.main, 'XTickMode','manual');
            cur_axis=get(h_axes.main);
            ii=1; start_subimage=orig_data.zspacing(1,ii); end_subimage=orig_data.zspacing(2,1); 
            for i=1:length(cur_axis.YTick)
                cur_pos=(cur_axis.YTick(i)/options.z_axis_sf-options.global_z_lim(1))/(options.global_z_lim(2)-options.global_z_lim(1));
                while (cur_pos > end_subimage)
                    ii=ii+1;
                    start_subimage=orig_data.zspacing(1,ii);
                    end_subimage=orig_data.zspacing(2,ii);
                end
                if (cur_pos < start_subimage)
                    cur_val(i)=NaN;
                else
                    cur_val(i)=round(10*(((cur_pos - start_subimage) / (end_subimage - start_subimage) * (options.global_z_lim(2)-options.global_z_lim(1))+options.global_z_lim(1)) * options.z_axis_sf))/10.0;
                end
            end
            iNaN=find(isnan(cur_val));
            XTickLabel1=num2cell(cur_val.');
            if (length(iNaN)>0)
                try
                    XTickLabel1{iNaN}='';
                catch
                
                end
            end
            set(h_axes.main, 'YTickLabel',XTickLabel1);
            set(h_axes.main, 'YTickMode','manual')
        end
        
        if (isfield(options,'viewnames_active') && options.viewnames_active>0 && (isfield(options,'pvalues_active') && options.pvalues_active<2  || ~isfield(options,'pvalues_active')))
            ii=0; %orig_data.views_start;
            for i=1:size(orig_data.zspacing,2)
                cur_z = ((0.05 * (orig_data.zspacing(2,i) - orig_data.zspacing(1,i))+orig_data.zspacing(1,i))* (options.global_z_lim(2)-options.global_z_lim(1))+options.global_z_lim(1)) * options.z_axis_sf; % 5% into image from top
                for j=1:size(orig_data.xspacing,2)
                    ii=ii+1;
                    if (ii> length(orig_data.view_names))
                        continue;
                    end
                    cur_x = ((0.05 * (orig_data.xspacing(2,j) - orig_data.xspacing(1,j))+orig_data.xspacing(1,j))* (options.global_x_lim(2)-options.global_x_lim(1))+options.global_x_lim(1)) * options.x_axis_sf; % 5% into image from left
                    text(h_axes.main,cur_x,cur_z,orig_data.view_names{ii},'Color','red','BackgroundColor','w','VerticalAlignment','top','FontSize',12,'Tag','text1');
                end
            end
        end
        
        if strcmp(options.plotwhat, 'arg')
            a = linspace(0, 2 * pi, 100)';
            if options.monochrome
                cmap = 0.5 * (1 + cos(a)) * [1,1,1];
            else
                cmap = (1 + [cos(a), cos(a + 2 * pi / 3), cos(a + 4 * pi / 3)]) / 2;
            end
        else
            if options.monochrome
                cmap = linspace(0,1,100)' * [1,1,1];
            else
                cmap = jet;
            end
        end
        f = ancestor(h_plot_objects.image, 'figure');
        if isfield(options,'pvalues_active') && options.pvalues_active>0
            set(f, 'colormap', flipud(cmap));
        else
            set(f, 'colormap', cmap);
        end
        fn_update_cursor;
        axis(h_axes.main, 'ij');
        set(h_axes.main, 'ButtonDownFcn', @cb_button_down_main);
        %plot geometric features, such as array
        delete(findall(h_axes.main,'Tag','geom'))
        if isfield(orig_data, 'geom') && (isfield(options,'pvalues_active') && options.pvalues_active<2  || ~isfield(options,'pvalues_active'))
            if isfield(orig_data.geom, 'array')
                line(orig_data.geom.array.x * options.x_axis_sf, orig_data.geom.array.z * options.z_axis_sf, ...
                    'Color', config.array_el_edge_color, ...
                    'Parent', h_axes.main, ...
                    'Tag', 'geom');
            end
            if isfield(orig_data.geom, 'lines')
                for ii = 1:length(orig_data.geom.lines)
                    if (isfield(orig_data.geom.lines(ii),'color')) %% Added color option
                        line(orig_data.geom.lines(ii).x * options.x_axis_sf, orig_data.geom.lines(ii).z * options.z_axis_sf, ...
                            'LineStyle', orig_data.geom.lines(ii).style, ...
                            'Color', orig_data.geom.lines(ii).color, ...
                            'Parent', h_axes.main, ...
                            'Tag', 'geom');

                    else
                        line(orig_data.geom.lines(ii).x * options.x_axis_sf, orig_data.geom.lines(ii).z * options.z_axis_sf, ...
                            'LineStyle', orig_data.geom.lines(ii).style, ...
                            'Color', 'w', ...
                            'Parent', h_axes.main, ...
                            'Tag', 'geom');
                    end
                end
            end
        end
        
        if (isfield(options,'select') && ~isempty(options.select))
            if (isfield(orig_data,'combined_plot') && orig_data.combined_plot>0 && (isfield(options,'pvalues_active') && options.pvalues_active<2  || ~isfield(options,'pvalues_active'))) %% Added combined plot tick control RB 2018/12/17
                % Update south graph
                % Update west graph
            else
                % Update south graph
                if size(options.select, 1)
                    ii = [1:length(plot_data.z)];
                    jj = interp1(plot_data.z, ii, options.select(1,2), 'nearest', 'extrap');
                    tmp = plot_data.f(jj, :);
                    [plot_val, limits, options.scale_mode] = fn_convert_to_plot_val(tmp, options.plotwhat, options.scale_mode, options.db_range, options.max_val, options.norm_val);
                    plot(h_axes.south, plot_data.x * options.x_axis_sf, plot_val, 'r');
                    ylim(h_axes.south, limits);
                end;
                
                % Update west graph
                if size(options.select, 1)
                    ii = [1:length(plot_data.x)];
                    jj = interp1(plot_data.x, ii, options.select(1,1), 'nearest', 'extrap');
                    tmp = plot_data.f(:, jj);
                    [plot_val, limits, options.scale_mode] = fn_convert_to_plot_val(tmp, options.plotwhat, options.scale_mode, options.db_range, options.max_val, options.norm_val);
                    plot(h_axes.west, plot_val, plot_data.z * options.z_axis_sf, 'r');
                    xlim(h_axes.west, limits);
                end;
            end
        end
        
        data_has_been_plotted = 1;
        
        %show colorbar
        fn_colorbar;
        
        %Update string
        if isfield(orig_data, 'text')
            str = [pointer_str, orig_data.text];
        else
            str = pointer_str;
        end
        
        set(h_plot_objects.selection_text, 'String', str);
        set(h_plot_objects.range_text, 'String', range_str);
        
        %         fn_set_sliders;
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

    function fn_zoom_out
        options.x_lim = options.global_x_lim;
        options.z_lim = options.global_z_lim;
        axis(h_axes.main, [options.global_x_lim * options.x_axis_sf, options.global_z_lim * options.z_axis_sf]);
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
        if (~isfield(options,'select') || isempty(options.select))
            return;
        end
        if fn_get_control_status(h_toolbar, 'cursor.point')
            set(h_plot_objects.region, 'visible', 'off');
            if (isfield(orig_data,'combined_plot') && orig_data.combined_plot > 0 && (isfield(options,'pvalues_active') && options.pvalues_active<2  || ~isfield(options,'pvalues_active')))
                if size(options.select, 1)
                    zLoc=(options.select(1,2)-options.global_z_lim(1)) / (options.global_z_lim(2)-options.global_z_lim(1));
                    xLoc=(options.select(1,1)-options.global_x_lim(1))/ (options.global_x_lim(2)-options.global_x_lim(1));
                    %Convert to actual xLoc and zLoc for combined plot
                    ii=1; start_subimage=orig_data.xspacing(1,ii); end_subimage=orig_data.xspacing(2,ii);
                    while (xLoc > end_subimage)
                        ii=ii+1;
                        if (ii > size(orig_data.xspacing,2))
                            return
                        end
                        start_subimage=orig_data.xspacing(1,ii);
                        end_subimage=orig_data.xspacing(2,ii);
                    end
                    if (xLoc < start_subimage)
                        xLoc=NaN;
                        xLoc2=NaN;
                    else
                        xLoc = (xLoc - start_subimage);
                        xLoc2= (xLoc / (end_subimage - start_subimage)* (options.global_x_lim(2)-options.global_x_lim(1)) + options.global_x_lim(1))* options.x_axis_sf;
                    end
                    ii=1; start_subimage=orig_data.zspacing(1,ii); end_subimage=orig_data.zspacing(2,ii);
                    while (zLoc > end_subimage)
                        ii=ii+1;
                        if (ii > size(orig_data.zspacing,2))
                            return
                        end
                        start_subimage=orig_data.zspacing(1,ii);
                        end_subimage=orig_data.zspacing(2,ii);
                    end
                    if (zLoc < start_subimage)
                        zLoc=NaN;
                        zLoc2=NaN;
                    else
                        zLoc=(zLoc - start_subimage) ;
                        zLoc2= (zLoc/ (end_subimage - start_subimage)* (options.global_z_lim(2)-options.global_z_lim(1)) + options.global_z_lim(1)) * options.z_axis_sf;
                    end
                    % Need to build cross-hair plot lines which cover all sub images
                    % Z-crosshair line
                    lx=c(2)-c(1); a=[c(1)-0.05*lx, c(2)+0.05*lx]; x_path=[]; b=[a(2) a(1)];z_path=[];
                    lz=c(4)-c(3); az=[c(3)-0.05*lz, c(4)+0.05*lz];  bz=[az(2) az(1)];
                    for i=1:size(orig_data.zspacing,2)
                        if (mod(i,2)>0)
                            x_path=[x_path a];
                        else
                            x_path=[x_path b];
                        end
                        a1=(options.global_z_lim(1)+(orig_data.zspacing(1,i)+zLoc)* (options.global_z_lim(2)-options.global_z_lim(1))) * options.z_axis_sf;
                        z_path=[z_path a1*ones(1,2)];
                    end
                    set(h_plot_objects.z_crosshair, 'XData', x_path, 'YData', z_path , 'Visible', 'On');
                    % X-crosshair line
                    z_path=[]; x_path=[];
                    for i=1:size(orig_data.xspacing,2)
                        if (mod(i,2)>0)
                            z_path=[z_path az];
                        else
                            z_path=[z_path bz];
                        end
                        a1=(options.global_x_lim(1)+(orig_data.xspacing(1,i)+xLoc)* (options.global_x_lim(2)-options.global_x_lim(1))) * options.x_axis_sf;
                        x_path=[x_path a1*ones(1,2)];
                    end
                    set(h_plot_objects.x_crosshair, 'XData', x_path, 'YData', z_path , 'Visible', 'On');
                    s1(1,1)=xLoc2/options.x_axis_sf;
                    s1(1,2)=zLoc2/options.z_axis_sf;
                    options.combined_select_index=s1;
                    vals1=fn_get_value_at_point_combined(orig_data, s1);
                    vals2=vals1(~isnan(vals1));
                    h=figure(100); clf;
                    h.Name='Extracted Data';
                    h.NumberTitle='off';
                    switch options.scale_mode
                        case 'linear'
                            max_val=max(abs(vals2) / options.norm_val * 100);
                            bar(abs(vals1) / options.norm_val * 100);
                            val_str = sprintf('%.3f %% ', abs(vals2) / options.norm_val * 100);
                            ylabel('Amplitude %%');
                        case 'log'
                            max_val=max(20*log10(abs(vals2) / options.norm_val));
                            bar_vals=20*log10(abs(vals1) / options.norm_val);
                            if (isfield(options,'levelling_active') && options.levelling_active>0 && isfield(orig_data,'mask'))
                                max_val=max(max_val+0.5,13);
                                bar(bar_vals);
                                hold on
                                plot([0.5 length(vals1)+0.5],[12 12],'--r');
                                axis([0.5 length(vals1)+0.5 min(20*log10(abs(vals2) / options.norm_val))-0.5 max_val]);
                                ylabel('Amplitude Relative to RMS (dB)');
                            else
                                if (isfield(orig_data,'rms')) 
                                    L={'TFM Intensity','Noise RMS'};
                                    rms_diff=20*log10(orig_data.rms / options.norm_val).';
                                    bar_vals=[bar_vals rms_diff];
                                    h=bar(bar_vals);
                                    legend(h,L,'Location','southoutside','Orientation','horizontal');
                                    legend('boxoff')
                                else
                                    bar(bar_vals);
                                end
                                ylabel('Amplitude (dB)');
                            end
                            val_str = sprintf('%.1f dB ', 20*log10(abs(vals2) / options.norm_val));
                            
                    end;
                    xlabel('View');
                    set(gca,'xtick',1:length(orig_data.view_names));
                    set(gca,'xticklabel',orig_data.view_names)
                    
                    pos_str = sprintf('(%.2f, %.2f)', s1(1,1) * options.x_axis_sf, s1(1,2) * options.z_axis_sf);
                    title(['[X,Z] = ',pos_str]);
                    %pointer_str = {pos_str, val_str};
                    pointer_str = {pos_str};
                end;
            else
                if size(options.select, 1)
                    set(h_plot_objects.x_crosshair, 'XData', [c(1), c(2)], 'YData', ones(1,2) * options.select(1,2) * options.z_axis_sf, 'Visible', 'On');
                    set(h_plot_objects.z_crosshair, 'XData', ones(1,2) * options.select(1,1) * options.x_axis_sf, 'YData', [c(3), c(4)] , 'Visible', 'On');
                    switch options.scale_mode
                        case 'linear'
                            val_str = sprintf('%.3f %%', abs(fn_get_value_at_point(plot_data, options.select)) / options.norm_val * 100);
                        case 'log'
                            val_str = sprintf('%.1f dB', 20*log10(abs(fn_get_value_at_point(plot_data, options.select)) / options.norm_val));
                    end;
                    pos_str = sprintf('(%.2f, %.2f)', options.select(1,1) * options.x_axis_sf, options.select(1,2) * options.z_axis_sf);
                    pointer_str = {pos_str, val_str};
                end;
            end
        else
            try
                close(100);
            catch
            
            end
        end;
        if fn_get_control_status(h_toolbar, 'cursor.region')
            if size(options.select, 1) == 0
                set(h_plot_objects.x_crosshair, 'visible', 'off');
                set(h_plot_objects.z_crosshair, 'visible', 'off');
                set(h_plot_objects.region, 'visible', 'off');
            else
                if size(options.select, 1) == 1
                    set(h_plot_objects.x_crosshair, 'XData', [c(1), c(2)], 'YData', ones(1,2) * options.select(1,2) * options.z_axis_sf, 'Visible', 'On');
                    set(h_plot_objects.z_crosshair, 'XData', ones(1,2) * options.select(1,1) * options.x_axis_sf, 'YData', [c(3), c(4)] , 'Visible', 'On');
                    x = options.select(1,1) * options.x_axis_sf;
                    y = options.select(1,2) * options.z_axis_sf;
                    set(h_plot_objects.region, 'XData', x, 'YData', y, 'visible', 'on');
                    pointer_str = sprintf('(%.2f, %.2f)', options.select(1,1) * options.x_axis_sf, options.select(1,2) * options.z_axis_sf);
                else
                    x = [options.select(1,1), options.select(2,1), options.select(2,1), options.select(1,1), options.select(1,1)] * options.x_axis_sf;
                    y = [options.select(1,2), options.select(1,2), options.select(2,2), options.select(2,2), options.select(1,2)] * options.z_axis_sf;
                    set(h_plot_objects.region, 'XData', x, 'YData', y, 'visible', 'on');
                    switch options.scale_mode
                        case 'linear'
                            val_str = [sprintf('%.3f %% (max); ', abs(fn_get_max_in_region(plot_data, options.select)) / options.norm_val * 100), ...
                                sprintf('%.3f %% (rms)', abs(fn_get_rms_in_region(plot_data, options.select)) / options.norm_val * 100)];
                        case 'log'
                            val_str = [sprintf('%.1f dB (max); ', 20*log10(abs(fn_get_max_in_region(plot_data, options.select)) / options.norm_val)), ...
                                sprintf('%.1f dB (rms)', 20*log10(abs(fn_get_rms_in_region(plot_data, options.select)) / options.norm_val))];
                    end
                    pos_str = sprintf('(%.2f, %.2f) to (%.2f, %.2f)', [options.select(1,1)' * options.x_axis_sf, options.select(1,2)' * options.z_axis_sf, options.select(2,1)' * options.x_axis_sf, options.select(2,2)' * options.z_axis_sf]);
                    pointer_str = {pos_str, val_str};
                end;
            end;
        end;
        if ~fn_get_control_status(h_toolbar, 'cursor.point') & ~fn_get_control_status(h_toolbar, 'cursor.region')
            %Remove selected point, if it exists
            if isfield(options,'select')
                options=rmfield(options,'select');
            end
            set(h_plot_objects.x_crosshair, 'visible', 'off');
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
    otherwise
        % do nothing
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
i2 = interp1(data.z, [1:length(data.z)], pt(2), 'nearest');
if i1 >= 1 & i1 <= size(data.f, 2) & i2 >= 1 & i2 <= size(data.f, 1)
    if (isfield(data,'pvalues_active') && data.pvalues_active>1)
        val= data.f2(i2, i1);
    else
        val = data.f(i2, i1);
    end
else
    val = 0;
end
end

function val = fn_get_value_at_point_combined(data, pt)
i1 = interp1(data.x, [1:length(data.x)], pt(1), 'nearest');
i2 = interp1(data.z, [1:length(data.z)], pt(2), 'nearest');
if (i1 < 1 || i1 > length(data.x) || isnan(i1) || i2 < 1 || i2 > length(data.z) || isnan(i2))
    val = 0;
    return;
end
if (isfield(data,'combined_plot') && data.combined_plot > 0 && (isfield(data,'pvalues_active') && data.pvalues_active<2  || ~isfield(data,'pvalues_active')))
    val=zeros(length(data.view_names),1);
    zs=round(data.zspacing*data.zfull);
    xs=round(data.xspacing*data.xfull);
    ii=0;
    for j=1:size(data.zspacing,2)
        jc=zs(1,j)+i2;
        for i=1:size(data.xspacing,2)
            ii=ii+1;
            if (ii>length(data.view_names))
                return;
            end
            ic=xs(1,i)+i1;
            val(ii)=data.f(jc, ic);
        end
    end   
else
    %if i1 >= 1 & i1 <= size(data.f, 2) & i2 >= 1 & i2 <= size(data.f, 1)
    if (isfield(options,'pvalues_active') && data.pvalues_active>1)
        val= data.f2(i2, i1);
    else
        val = data.f(i2, i1);
    end
    %end
end

end

function val = fn_get_rms_in_region(data, reg)
i1 = find(data.x >= min(reg(:, 1)) & data.x <= max(reg(:, 1)));
i2 = find(data.z >= min(reg(:, 2)) & data.z <= max(reg(:, 2)));
if min(i1) >= 1 & max(i1) <= size(data.f, 2) & min(i2) >= 1 & max(i2) <= size(data.f, 1)
    val = data.f(i2, i1);
else
    val = 0;
end
val = sqrt(mean(val(:) .* conj(val(:))));
end

function val = fn_get_max_in_region(data, reg)
i1 = find(data.x >= min(reg(:, 1)) & data.x <= max(reg(:, 1)));
i2 = find(data.z >= min(reg(:, 2)) & data.z <= max(reg(:, 2)));
if min(i1) >= 1 & max(i1) <= size(data.f, 2) & min(i2) >= 1 & max(i2) <= size(data.f, 1)
    val = data.f(i2, i1);
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