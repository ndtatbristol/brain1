function [fig_handle, h_fn_update_data, h_fn_get_options, h_fn_set_options] = gui_process_window(fn_imaging_process, fn_close, exp_data, start_in_folder, use_gpu, data_folder)
%this generates a generic processing window (i.e. with algorithm control
%and a display panel and returns handles to the update function (call as
%fn_update_data and a function to return the current options settings
%for the process to enable these to be recorded (call as options -
%fn_get_options)
%exp_data is passed to enable initial options to be set if they use
%exp_data parameters in function arguments

config = fn_get_config;

fig_handle = [];
h_fn_update_data = [];
h_fn_get_options = [];
h_fn_set_options = [];
%check if process function exists and if not return nothing (necessary for
%when Brain loads a previous setup (and consequently when it starts as well
%as last_setup is loaded at startup)
tmp = functions(fn_imaging_process);
if isempty(tmp.file)
    return
end

paused = 0;

info = fn_imaging_process(exp_data, [], 'return_info_only');

%add final option to all processes to enable/disable GPU with default a/c to how use_gpu is set
info.options_info.use_gpu_if_available.label = 'Use GPU if available';
info.options_info.use_gpu_if_available.type = 'bool';
info.options_info.use_gpu_if_available.constraint = {'On', 'Off'};
if use_gpu
    info.options_info.use_gpu_if_available.default = 1;
else
    info.options_info.use_gpu_if_available.default = 0;
end

fn = fieldnames(info.options_info);
process_options = [];
for ii = 1:length(fn)
    process_options = setfield(process_options, fn{ii}, getfield(info.options_info, fn{ii}, 'default'));
end

fn_display = info.fn_display;
display_options = info.display_options;
input_field_info = info.options_info;
process_name = info.name;

process_options.fn_process = fn_imaging_process;
%display_options = [];
input_field_info = info.options_info;

%variable holding actual displayed data
display_data = [];

%find available analysis tools
available_analysis = fn_get_available_analysis({config.files.analysis_path, fullfile(start_in_folder, config.files.local_brain_path, config.files.analysis_path)});

%load icons
icons = [];
load(config.files.icon_file);

%create the figure
fig_handle = figure(...
    'MenuBar', 'none', ...
    'NumberTitle', 'off', ...
    'ToolBar', 'None', ...
    'Units', 'normalized', ...
    'Name', ['Imaging: ', process_name], ...
    'DeleteFcn', @fn_close_window ...
    );
panel_colour = get(gcf, 'Color');

input_panel_rel_pos = [0, 0, 0.25, 1];
plot_panel_rel_pos = [0.25, 0, 0.75, 1];

%input parameter panel
h_input_panel = uipanel('BorderType','none',...
    'BackgroundColor', get(gcf, 'Color'),...
    'Units', 'normalized',...
    'Position', input_panel_rel_pos,...
    'Parent', fig_handle);

[h_input_table, fn_get_process_options, fn_set_process_options, fn_set_content] = gui_options_table(h_input_panel, [0, 0, 1, 1], 'normalized', @fn_options_changed);
% keyboard
fn_set_content(input_field_info);
% process_options = fn_get_process_options([]);
% process_options.fn_process = fn_imaging_process;
%plot panel
left_panel_rel_pos = [0.3, 0, 0.7, 1];
h_panel = uipanel('BorderType','none',...
    'BackgroundColor', panel_colour,...
    'Units', 'normalized',...
    'Position', plot_panel_rel_pos,...
    'Parent', fig_handle);

h_toolbar = uitoolbar(fig_handle);
h_pause = uitoggletool(h_toolbar, ...
                        'CData', fn_get_cdata_for_named_icon(icons, 'stop'), ...
                        'TooltipString', 'Pause', ...
                        'HandleVisibility', 'Off', ...
                        'State', 'On', ...
                        'OnCallback', @cb_play, ...
                        'OffCallback', @cb_pause);
uipushtool(h_toolbar, ...
                        'CData', fn_get_cdata_for_named_icon(icons, 'analyse'), ...
                        'TooltipString', 'Analyse image', ...
                        'HandleVisibility', 'Off', ...
                        'ClickedCallback', @cb_analysis);
                    
uipushtool(h_toolbar, ...
                        'CData', fn_get_cdata_for_named_icon(icons, 'Standard.SaveFigure'), ...
                        'TooltipString', 'Save image', ...
                        'HandleVisibility', 'Off', ...
                        'ClickedCallback', @cb_save);

%turn the panel into a plot panel
[fn_update_display, fn_get_display_options, fn_set_display_options] = fn_display(h_panel, h_toolbar);
fn_set_display_options(display_options)
%return handles
h_fn_update_data = @fn_update_data;
h_fn_get_options = @fn_get_options;
h_fn_set_options = @fn_set_options;

    function cb_analysis(a, b)
        if isempty(exp_data) | isempty(display_data)
            return
        end
        for ii = 1:length(available_analysis)
            proc_names{ii} = available_analysis(ii).name;
        end
        [jj, ok] = listdlg('ListString', proc_names, 'SelectionMode', 'single');
        if ~ok
            return;
        end
        disp_ops = fn_get_display_options();
        proc_ops = fn_get_process_options();
        try
            available_analysis(jj).fn_process(exp_data, display_data, disp_ops, proc_ops, fn_set_process_options, @fn_options_changed);
        catch
            available_analysis(jj).fn_process(exp_data, display_data, disp_ops); %earlier versions!
        end
    end

    function cb_play(a, b)
        set(h_pause, 'CData', fn_get_cdata_for_named_icon(icons, 'stop'));
        paused = 0;
    end

    function cb_pause(a, b)
        set(h_pause, 'CData', fn_get_cdata_for_named_icon(icons, 'play'));
        paused = 1;
    end

    function cb_save(a, b)
        filter{1,1} = '*.fig'; filter{1,2} = 'Matlab figure (*.fig)';
        filter{2,1} = '*.png'; filter{2,2} = 'Portable Network Graphics (*.png)';
        filter{3,1} = '*.jpg'; filter{3,2} = 'JPEG (*.jpg)';
        filter{4,1} = '*.eps'; filter{4,2} = 'EPS level 1 (*.eps)';
        if ~isempty(display_data)
            filter{5,1} = '*.mat'; filter{5,2} = 'Raw image data (*.mat)';
        end
        [fname, data_folder, filterindex] = uiputfile(filter, 'Save', [data_folder, filesep]);
        if (fname == 0) % Added to catch Save Dialog exiting without a specified filename (e.g. cancel button hit)
            %disp('No filename specified. Skipping')
            return;
        end
        if filterindex < 5
            saveas(fig_handle, fullfile(data_folder, fname));
        else
            save(fullfile(data_folder, fname), 'display_data');
        end
    end

    function fn_update_data(new_exp_data, mode)
        if ~paused
            exp_data = new_exp_data;
            fn_process_and_display(mode);
        end
    end

    function fn_process_and_display(mode)
        if ~isempty(exp_data)
            switch mode
                case 'recalc_and_process'
                    [display_data, process_options] = fn_imaging_process(exp_data, process_options, 'recalc_and_process');
                case 'process_only'
                    if (isfield(process_options, 'surface_type') && strcmp(process_options.surface_type, '|M|easured')>0)
                        [display_data, process_options] = fn_imaging_process(exp_data, process_options, 'process_only');
                    else
                        display_data = fn_imaging_process(exp_data, process_options, 'process_only');
                    end               
            end
            if isempty(display_data)
                return;
            end
            fn_set_process_options(process_options); % To update variables in table after calculation (if required)
            fn_update_display(display_data);
        end;
    end;

    function fn_close_window(src, ev)
        %call the function passed as argument
        set(fig_handle, 'Visible', 'Off');
        fn_close(src);
    end

    function [proc_ops, disp_ops] = fn_get_options(dummy)
        proc_ops = process_options;
        disp_ops = fn_get_display_options([]);
    end

    function fn_set_options(ops)
        if isfield(ops, 'position')
            set(fig_handle, 'Position', ops.position);
        end;
        if isfield(ops, 'display_options')
            display_options = ops.display_options;
            fn_set_display_options(display_options);
        end
        if isfield(ops, 'process_options')
            process_options = ops.process_options;
            fn_set_process_options(process_options);
        end;
    end

    function  fn_options_changed(ops)
        old_options = process_options;
        process_options = ops;
        process_options.fn_process = old_options.fn_process;
        process_options.options_changed = 1;
        try
            [display_data, process_options] = fn_imaging_process(exp_data, process_options, 'recalc_and_process');
        catch exc
            if strcmpi(exc.identifier, 'MATLAB:nomem')
                warndlg('Insufficient memory','Error');
            else
                warndlg(exc.message,'Error');
            end
            process_options = old_options;
            [display_data, process_options] = fn_imaging_process(exp_data, process_options, 'recalc_and_process');
        end
        fn_set_process_options(process_options);
        fn_update_display(display_data);
    end

end