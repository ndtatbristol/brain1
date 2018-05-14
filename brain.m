function brain(varargin)
%This is the function for the main window of the GUI and handles data
%acquisition, exp_data loading and saving, and display of raw data. It also
%contains the menu that enables processing windows to be launched,
%depending on the availability of processing functions

%The first optional argument is the 'local_folder'; if not specified or [],
%the current folder is used.

%The second optional argument is the setup file to use; if not specified
%the setup in the last_setup file will be used if found, if not the default
%setup is used. This must be either a file in the current directory or an
%absolute path

%close all windows etc. (but don't do a 'clear' as this causes problems
%with instrument DLLs)
evalin('base','close all, clear all')
close all;
clc;
global SYSTEM_TYPES
%first set of declarations create these as global variables on the
%workspace, while the second set make them available in the function
evalin('base','global Trans')
evalin('base','global Resource')
evalin('base','global TW')
evalin('base','global TX')
evalin('base','global Event')
evalin('base','global Receive')
evalin('base','global SeqControl')
evalin('base','global TGC')
evalin('base','global VDASupdates')
evalin('base','global VDAS')
evalin('base','global Control')
evalin('base','global VSX_Control')
evalin('base','global RcvProfile')
evalin('base','global TPC')

global Trans
global Resource
global TW
global TX
global Event
global Receive
global SeqControl
global TGC
global VDASupdates
global VDAS
global Control
global VSX_Control
global array
global ph_velocity
global RcvProfile
global TPC

%make sure no callbacks executed during setup
callbacks_disabled = 1;

if strcmp('8.1.0.604 (R2013a)',version)
    msgid ='parallel:gpu:array:FunctionToBeRemoved';
    warning('OFF',msgid);
end

a=ver('matlab');
SYSTEM_TYPES.mat_ver=str2num(a.Version);

if isdeployed
    disp('Main function brain.m starting');
end

if ~isdeployed
    %find current BRAIN folder if running in Matlab
    tmp = dbstack;
    brain_folder = fileparts(which(tmp.name));
else
    %not needed if running in Standalone
    brain_folder = '';
end

%Record current folder (only needed if running in Matlab so it can be
%restored when brain exits
if ~isdeployed
    current_dir_when_brain_started = pwd;
else
    current_dir_when_brain_started = pwd;
end

if ~isdeployed
    %if running in Matlab change current directory to location of the current function - so that all relative paths work
    try
        cd(brain_folder);
    catch
        disp(['Could not change directory (', brain_folder, ')']);
    end
end

%load configuration data
config = fn_get_config;
local_folder = current_dir_when_brain_started;
fn_prepare_brain_folders(local_folder, brain_folder, config);
setup = config.default_setup;
setup.gpu=1;


disp(['Brain folder: ', brain_folder]);
disp(['Local folder: ', local_folder]);
disp(['Current folder: ', pwd]);

%GPU
if nargin >= 1
    use_gpu = varargin{1};
    if isdeployed
        if strcmp(use_gpu, '-nogpu')
            use_gpu = 0;
        else
            use_gpu = 1;
        end
    end
else
    use_gpu = 1;
end

%actually test for GPU
use_gpu = use_gpu & fn_test_if_gpu_present_and_working(1);

% %following is bodge - should put these GPU files library
% if use_gpu
%     if ~isdeployed
%         addpath('GPU');
%     else
%         %??
%     end
% end

%FOLLOWING IS OBSOLETE (29/6/12) but may be useful at some point. Relates
%to arguments specifying setup file etc.
%Identify local folder (either folder where run from or first optional
%argument - this is same in both Matlab and Standalone versions
% if nargin >= 1 && ~isempty(varargin{1}) &&(exist(varargin{1}) == 7)
%     local_folder = varargin{1};
% else
%     local_folder = current_dir_when_brain_started;
% end

%Identify initial setup file (either last setup file in local folder
%structure of second optional argument - this is same in both Matlab and
%Standalone versions
% if nargin >= 2 && ~isempty(varargin{2})
%     setup_file = varargin{2};
% else
%     setup_file = fullfile(local_folder, config.files.local_brain_path, config.files.setups_path, config.files.last_setup_file);
% end
%     setup_file = fullfile(local_folder, config.files.local_brain_path, config.files.setups_path, config.files.last_setup_file);

%Prepare local folder structure if they don't exist already and brian is
%not being run from master folder

%load last setup or, if not found, default setup
% if exist(setup_file) == 2 & config.general.load_last_setup
%     load(setup_file, '-mat');
% else
% end

if ~isfield(setup, 'data_folder')
    setup.data_folder = local_folder;
end
if ~isfield(setup, 'setup_folder')
    setup.setup_folder = local_folder;
end

%load icons
icons = [];
load(config.files.icon_file);

%the master copy of exp_data is in this function and is the one either
%loaded from file or acquired from an instrument. use exp_data = [] to
%disable the displays etc.
exp_data = setup.last_exp_data;

%inst_options is the control structure for the instrument
instr_options = [];

%find available processes
available_imaging = fn_get_available_imaging({config.files.imaging_path, fullfile(local_folder, config.files.local_brain_path, config.files.imaging_path)});

%find available instruments
available_instruments = fn_get_available_instruments(config.files.instruments_path);
%determine which if any is verasonics
for qq=1:length(available_instruments)
   is_ver(qq)=strcmp(available_instruments(qq).instr_info.name,'Verasonics');
end
setup.verasonic=find(is_ver);

%get list of available array files
available_arrays = fn_get_available_arrays({config.files.arrays_path, fullfile(local_folder, config.files.local_brain_path, config.files.arrays_path)});

%get list of available material files
available_materials = fn_get_available_materials({config.files.materials_path, fullfile(local_folder, config.files.local_brain_path, config.files.materials_path)});

fn_instr_connect = [];
fn_instr_disconnect = [];
fn_instr_reset = [];
fn_instr_acquire = [];
fn_send_instr_options = [];

%various variables to initialise
main_window_closing = 0;
recalc_focal_laws = 0;
time_of_last_frame = [];
if isempty(exp_data)
    data_is_from_file = 0;
else
    data_is_from_file = 1;
end
fps_string = '';

window_title = [config.main_win.title, sprintf(' (%.2f)', config.main_win.version)];
if use_gpu
    window_title = [window_title, ' GPU'];
    config.main_win.gpu=1;
else
    window_title = [window_title, ' no GPU'];
    config.main_win.gpu=0;
end

%create the window
main.handle = figure('MenuBar', 'none', ...
    'Color', config.general.window_bg_color, ...
    'NumberTitle', 'off', ...
    'Name',  window_title, ...
    'Units', 'Normalized', ...
    'OuterPosition', setup.main_win.non_maximised_position, ...
    'DeleteFcn', @fn_close_main_window ...
    );
% jFrame = get(handle(main.handle),'JavaFrame');
% if isfield(setup, 'main_win.maximised')
%     jFrame.setMaximized(setup.main_win.maximised);
% end

%run the setup functions
fn_initialisation;
fn_update;
callbacks_disabled = 0;
%--------------------------------------------------------------------------

    function fn_initialisation
        %this first setup function is only called when application is first
        %executed and it creates all the buttons etc.
        
        %set up the toolbar
        main.toolbar.handle = uitoolbar(main.handle);
        main.control(1).handle = uipushtool(main.toolbar.handle, ...
            'CData', fn_get_cdata_for_named_icon(icons, 'Standard.SaveFigure'), ...
            'TooltipString', 'Save set-up or data', ...
            'Tag', 'save');
        main.control(2).handle = uipushtool(main.toolbar.handle, ...
            'CData', fn_get_cdata_for_named_icon(icons, 'Standard.FileOpen'), ...
            'TooltipString', 'Load set-up or data', ...
            'Tag', 'load');
        main.control(3).handle = uipushtool(main.toolbar.handle, ...
            'CData', fn_get_cdata_for_named_icon(icons, 'new-image'), ...
            'TooltipString', 'New process', ...
            'Tag', 'new');
        for ii = 1:length(main.control)
            set(main.control(ii).handle, 'HandleVisibility', 'Off', 'ClickedCallback', @cb_control);
        end;
        
        %the instrument status panel
        main.status_panel.handle = uipanel('BorderType', config.main_win.status_panel.border_type,...
            'BackgroundColor', get(gcf, 'Color'),...
            'Units', 'normalized',...
            'Position', config.main_win.status_panel.position,...
            'Parent', main.handle);
        h = axes('Parent', main.status_panel.handle, 'OuterPosition', config.main_win.status_panel.axes_position, 'Color', config.general.window_bg_color);
        set(h, 'Fontsize', 8);
        main.status_image.handle = imagesc(rand(10), 'Parent', h);
        set(get(h, 'XLabel'), 'String', 'Transmitter', 'Fontsize', config.main_win.status_panel.font_size);
        set(get(h, 'YLabel'), 'String', 'Receiver', 'Fontsize', config.main_win.status_panel.font_size);
        main.status_image.title = uicontrol('parent', main.status_panel.handle, ...
            'Units', 'Normalized', ...
            'Position', config.main_win.status_panel.title_position, ...
            'Style', 'text', ...
            'String', {' ', ' ', ' '}, ...
            'Fontsize', config.main_win.status_panel.font_size, ...
            'BackgroundColor', config.general.window_bg_color, ...
            'HorizontalAlignment', 'Left');
        
        %the instrument control panel
        main.instr_panel.handle = uipanel('BorderType', config.main_win.instr_panel.border_type,...
            'BackgroundColor', config.general.window_bg_color,...
            'Units', 'normalized',...
            'Position', config.main_win.instr_panel.position,...
            'Parent', main.handle);
        
        %instrument control table
        [main.instr_panel.h_table, main.instr_panel.fn_get_data, main.instr_panel.fn_set_data, main.instr_panel.fn_set_content] = ...
            gui_options_table(main.instr_panel.handle, [0, 0, 1, 1], 'normalized', @fn_instr_options_changed);
        
        main.instr_panel.h_instrument = uicontrol(main.instr_panel.handle, 'Style', 'PopUpMenu', 'Tag', 'instr_select', 'HandleVisibility', 'Off', 'Callback', @cb_control);
        main.instr_panel.h_array = uicontrol(main.instr_panel.handle, 'Style', 'PopUpMenu', 'Tag', 'array_select', 'HandleVisibility', 'Off', 'Callback', @cb_control);
        main.instr_panel.h_material = uicontrol(main.instr_panel.handle, 'Style', 'PopUpMenu', 'Tag', 'material_select', 'HandleVisibility', 'Off', 'Callback', @cb_control);
        
        fn_reset_listboxes;
        
        %instrument control buttons
        main.instr_panel.h_connect = uicontrol(main.instr_panel.handle, 'Style', 'ToggleButton', 'CData', fn_get_cdata_for_named_icon(icons, 'connect'), 'Tag', 'instr_connect', 'HandleVisibility', 'Off', 'Callback', @cb_control, 'Enable', 'Off');
        main.instr_panel.h_reset = uicontrol(main.instr_panel.handle, 'Style', 'PushButton', 'CData', fn_get_cdata_for_named_icon(icons, 'reset'), 'Tag', 'instr_reset', 'HandleVisibility', 'Off', 'Callback', @cb_control, 'Enable', 'Off');
        main.instr_panel.h_play = uicontrol(main.instr_panel.handle, 'Style', 'ToggleButton', 'CData', fn_get_cdata_for_named_icon(icons, 'play'), 'Tag', 'instr_play', 'HandleVisibility', 'Off', 'Callback', @cb_control, 'Enable', 'Off');
        main.instr_panel.h_play_once = uicontrol(main.instr_panel.handle, 'Style', 'PushButton', 'CData', fn_get_cdata_for_named_icon(icons, 'play-once'), 'Tag', 'instr_play_once', 'HandleVisibility', 'Off', 'Callback', @cb_control, 'Enable', 'Off');
        
        %         if isfield(setup, 'instr_options')
        %             main.instr_panel.fn_set_data(setup.instr_options);
        %         end
        
        %the plotting region for raw data
        main.raw_panel.handle = uipanel('BorderType', config.main_win.raw_data_panel.border_type,...
            'BackgroundColor', get(gcf, 'Color'),...
            'Units', 'normalized',...
            'Position', config.main_win.raw_data_panel.position,...
            'Parent', main.handle);
        setup.raw_display_options.axis_equal = 0;
        setup.raw_display_options.x_axis_sf = config.main_win.raw_data_panel.time_axis_scalefactor;
        setup.raw_display_options.z_axis_sf = config.main_win.raw_data_panel.vert_axis_scalefactor;
        setup.raw_display_options.interpolate = 0;
        setup.raw_display_options.x_label = config.main_win.raw_data_panel.time_axis_label;
        setup.raw_display_options.z_label = config.main_win.raw_data_panel.vert_axis_label;
        
        %add a standard 2D plot window to show the raw data
        [main.raw_panel.fn_update, main.raw_panel.fn_get_display_options, main.raw_panel.fn_set_display_options] = gui_2d_plot_panel(main.raw_panel.handle, main.toolbar.handle);
        main.raw_panel.fn_set_display_options(setup.raw_display_options);
        
        %call second part of function to deal with settings according to
        %setup
        
        set(main.handle, 'ResizeFcn', @fn_main_window_resize);
        fn_main_window_resize([],[]);
        fn_apply_setup;
    end

    function fn_apply_setup
        %apply setup settings to main window (need to check for everyfield
        %and implement default if not found for backward compatibility
        
        callbacks_disabled = 1;
         %select last material or 1 if not specified
        if isfield(setup, 'current_matl')
            fn_material_select(setup.current_matl);
        else
            fn_material_select(config.default_setup.current_matl);
        end
        
        %select last array or 1 if not specified
        if isfield(setup, 'current_array')
            fn_array_select(setup.current_array);
        else
            fn_array_select(config.default_setup.current_array);
        end
        
        %select last instrument or 1 if not specified
        if isfield(setup, 'current_instr')
            fn_instrument_select(setup.current_instr);
        else
            fn_instrument_select(config.default_setup.current_instr);
        end
        
        %fill up instrument control params
        if isfield(setup, 'instr_options')
            main.instr_panel.fn_set_data(setup.instr_options);
        end
              
        
        %create all defined processing windows
        if isfield(setup, 'gui_process_window')
            fig_exists = ones(length(setup.gui_process_window), 1);
            for ii = 1:length(setup.gui_process_window)
                [setup.gui_process_window(ii).figure_handle, ...
                    setup.gui_process_window(ii).fn_update_data_handle, ...
                    setup.gui_process_window(ii).fn_get_options_handle, ...
                    setup.gui_process_window(ii).fn_set_options_handle] = ...
                    gui_process_window(setup.gui_process_window(ii).process_options.fn_process, ...
                    @fn_display_window_closing, exp_data, local_folder, use_gpu, setup.data_folder);
                if ~isempty(setup.gui_process_window(ii).figure_handle)
                    fig_exists(ii) = 1;
                    setup.gui_process_window(ii).fn_set_options_handle(setup.gui_process_window(ii));
                else
                    fig_exists(ii) = 0;
                end
            end
            setup.gui_process_window(find(fig_exists == 0)) = [];
        end
        callbacks_disabled = 0;
    end

    function fn_main_window_resize(src, eventdata)
        p = getpixelposition(main.instr_panel.handle);
        
        %array
        row = 1;
        setpixelposition(main.instr_panel.h_array, [    1,                  p(4) - config.general.button_height_pixels * row + 1,       p(3),       config.general.button_height_pixels]);
        
        %material
        row = 2;
        setpixelposition(main.instr_panel.h_material,[    1,                  p(4) - config.general.button_height_pixels * row + 1,       p(3),       config.general.button_height_pixels]);
        
        %instrument select
        row = 3;
        setpixelposition(main.instr_panel.h_instrument, [1,                 p(4) - config.general.button_height_pixels * row + 1,   p(3),       config.general.button_height_pixels]);
        
        %instrument controls
        row = 4;
        setpixelposition(main.instr_panel.h_connect, [  1,                  p(4) - config.general.button_height_pixels * row + 1,   p(3) / 4,   config.general.button_height_pixels]);
        setpixelposition(main.instr_panel.h_reset, [    p(3) / 4 + 1,       p(4) - config.general.button_height_pixels * row + 1,   p(3) / 4,   config.general.button_height_pixels]);
        setpixelposition(main.instr_panel.h_play, [     p(3) / 4 * 2 + 1,   p(4) - config.general.button_height_pixels * row + 1,   p(3) / 4,   config.general.button_height_pixels]);
        setpixelposition(main.instr_panel.h_play_once, [p(3) / 4 * 3 + 1,   p(4) - config.general.button_height_pixels * row + 1,   p(3) / 4,   config.general.button_height_pixels]);
        
        %table fills rest
        setpixelposition(main.instr_panel.h_table, [1, 1, p(3), p(4) - config.general.button_height_pixels * row]);
    end

    function fn_display_window_closing(fh)
        if ~main_window_closing
            if isfield(setup, 'gui_process_window')
                for ii = 1:length(setup.gui_process_window)
                    try
                        if setup.gui_process_window(ii).figure_handle == fh
                            setup.gui_process_window(ii) = [];
                            break;
                        end
                    catch
                        %nesc to avoid problems of window closing mid loop
                    end
                end
            end
        end
    end

    function fn_reset_listboxes
        %Populates instrument, array and material list boxes (called at
        %startup and when an instrument is selected after a file has
        %previously been loaded
        fn_populate_instr_pulldown;
        fn_populate_arrays_pulldown;
        fn_populate_materials_pulldown;
    end

    function fn_populate_instr_pulldown
        %instrument pull-down
        for ii = 1:length(available_instruments)
            strs{ii} = available_instruments(ii).instr_info.name;
        end
        set(main.instr_panel.h_instrument, 'String', strs);
        clear strs;
    end

    function fn_populate_arrays_pulldown
        for ii = 1:length(available_arrays)
            strs{ii} = available_arrays(ii).name;
        end
        strs{length(strs) + 1} = 'Create new array';
        set(main.instr_panel.h_array, 'String', strs);
    end

    function fn_populate_materials_pulldown
        for ii = 1:length(available_materials)
            strs{ii} = available_materials(ii).name;
        end
        strs{length(strs) + 1} = 'Create new material';
        set(main.instr_panel.h_material, 'String', strs);
    end


    function fn_set_listboxes_to_match_exp_data
        %Sets instrument list box to new field ('Data from file'),
        %also adds and selects line in array and material list boxes
        %for data found in exp_data. Called when a previous file is loaded.
        str = get(main.instr_panel.h_instrument, 'String');
        str{end + 1} = 'Data from file';
        set(main.instr_panel.h_instrument, 'String', str, 'Value', length(str));
        
        str = get(main.instr_panel.h_material, 'String');
        if isfield(exp_data, 'vel_poly')
            tmp = ' (var)';
        else
            tmp = '';
        end;
        str{end + 1} = sprintf(['Velocity: %.1f m/s', tmp], exp_data.ph_velocity);
        set(main.instr_panel.h_material, 'String', str, 'Value', length(str));
        
        str = get(main.instr_panel.h_array, 'String');
        if isfield(exp_data.array, 'centre_freq')
            str{end + 1} = sprintf('Array: %i els, %.1f MHz', length(exp_data.array.el_xc), exp_data.array.centre_freq / 1e6);
        else
            str{end + 1} = sprintf('Array: %i els', length(exp_data.array.el_xc));
        end
        set(main.instr_panel.h_array, 'String', str, 'Value', length(str));
    end

    function cb_control(src, ev)
        %callback for all events - called functions follow in order
        if callbacks_disabled
            return
        end
        tag = get(src, 'Tag');
        switch(tag)
            case 'load'
                fn_load;
            case 'save'
                fn_save;
            case 'new'
                fn_new_process();
            case 'material_select'
                ii = get(main.instr_panel.h_material, 'value');
                fn_material_select(ii);
            case 'array_select'
                ii = get(main.instr_panel.h_array, 'value');
                fn_array_select(ii);                
            case 'instr_connect'
                array = fn_get_array;
                [ph_velocity, vel_poly] = fn_get_ph_velocity;
                fn_instrument_connect;
            case 'instr_reset'
                fn_instrument_reset;
            case 'instr_select'
                ii = get(main.instr_panel.h_instrument, 'value');
                if ii==setup.verasonic
                  array = fn_get_array;
                  available_instruments(setup.verasonic).instr_info.options_info.pulse_freq.default=array.centre_freq./1e6;
                end
                fn_instrument_select(ii);
            case 'instr_play_once'
                fn_play(1);
            case 'instr_play'
                fn_play(0);
            
        end
    end

    function fn_load
        [tmp_exp_data, tmp_options, exp_data_fname] = fn_load_file(setup.data_folder, fullfile(local_folder, config.files.local_brain_path, config.files.arrays_path));
        if ~isempty(exp_data_fname)
            setup.data_folder = fileparts(exp_data_fname);
        end
        if ~isempty(tmp_exp_data)
            %copy in the exp_data
            exp_data = tmp_exp_data;
            %make sure continuous acquisition stopped
            set(main.instr_panel.h_play, 'value', 0);
            %disconnect instrument
            fn_instrument_disconnect;
            
            fn_set_listboxes_to_match_exp_data;
            
            fn_update;
            data_is_from_file = 1;
            %             %disable array and material selection pull-downs
            %             set(main.instr_panel.h_material, 'Enable', 'Off');
            %             set(main.instr_panel.h_array, 'Enable', 'Off');
            %update display and any child windows
            %update setup
            %             setup.last_exp_data_file = tmp_options.last_exp_data_file;
            return;
        end
        if ~isempty(tmp_options)
            setup = tmp_options;
            fn_reset;
        end
    end

    function fn_save
        fn_update_setup_variable;
        fname = fn_save_file(exp_data, setup, setup.data_folder);
        if ~isempty(fname)
            setup.data_folder = fileparts(fname);
        end
    end

    function fn_new_process(dummy)
        %check data is there and all nesc fields are present
        if isempty(exp_data)
            warndlg('Load or acquire array data first','Warning')
            return;
        end;
        if ~isfield(exp_data, 'array')
            warndlg('Select array first','Warning')
            return;
        end;
        if ~isfield(exp_data, 'ph_velocity')
            warndlg('Select material first','Warning')
            return;
        end;
        %show a list box of available processes and let user select one
        for ii = 1:length(available_imaging)
            proc_names{ii} = available_imaging(ii).name;
        end;
        [jj, ok] = listdlg('ListString', proc_names, 'SelectionMode', 'single');
        if ~ok
            return;
        end;
        
        %add a new panel get a handle to the update function for the
        %display
        if isfield(setup, 'gui_process_window')
            ii = length(setup.gui_process_window) + 1;
        else
            ii = 1;
        end
        %         keyboard
        [setup.gui_process_window(ii).figure_handle, ...
            setup.gui_process_window(ii).fn_update_data_handle, ...
            setup.gui_process_window(ii).fn_get_options_handle, ...
            setup.gui_process_window(ii).fn_set_options_handle] = ...
            gui_process_window( ...
            available_imaging(jj).fn_process, ...
            @fn_display_window_closing, ...
            exp_data, local_folder, use_gpu, setup.data_folder);
        if ii > 1
            last_options = setup.gui_process_window(ii - 1).fn_get_options_handle();
            current_options = setup.gui_process_window(ii).fn_get_options_handle();
            new_options.process_options = fn_copy_common_process_options(last_options, current_options);
            %             keyboard
            setup.gui_process_window(ii).fn_set_options_handle(new_options);
        end
        setup.gui_process_window(ii).fn_update_data_handle(exp_data, 'recalc_and_process');
    end

    function new_options = fn_copy_common_process_options(last_options, current_options)
        current_fieldnames = fieldnames(current_options);
        last_fieldnames = fieldnames(last_options);
        common_fieldnames = intersect(current_fieldnames, last_fieldnames);
        new_options = current_options;
        for ii = 1:length(common_fieldnames)
            new_options = setfield(new_options, common_fieldnames{ii}, getfield(last_options, common_fieldnames{ii}));
        end
    end

    function fn_instrument_connect
        if get(main.instr_panel.h_connect, 'Value')
            set(main.status_image.title, 'String', ['Connecting to ', available_instruments(setup.current_instr).instr_info.name]);
            res = available_instruments(setup.current_instr).fn_instr_connect(main.instr_panel.fn_get_data([]));
            pause(config.main_win_instr_connect_delay);
        else
            set(main.status_image.title, 'String', ['Disconnecting from ', available_instruments(setup.current_instr).instr_info.name]);
            fn_instrument_disconnect;
            pause(config.main_win_instr_connect_delay);
            return;
        end
        if ~res
            fn_instrument_disconnect;
            warndlg('Failed to connect', 'Warning', 'Modal')
            return;
        end
        %enable the other instrument control buttons if connection made
        set(main.status_image.title, 'String', ['Connected to ', available_instruments(setup.current_instr).instr_info.name]);
        set(main.instr_panel.h_connect, 'Value', 1);
        set(main.instr_panel.h_reset, 'Enable', 'On');
        set(main.instr_panel.h_play, 'Enable', 'On');
        set(main.instr_panel.h_play_once, 'Enable', 'On');
    end

    function fn_instrument_disconnect
        available_instruments(setup.current_instr).fn_instr_disconnect();
        %enable the other instrument control buttons if connection made
        set(main.instr_panel.h_connect, 'Value', 0);
        set(main.instr_panel.h_reset, 'Enable', 'Off');
        set(main.instr_panel.h_play, 'Enable', 'Off');
        set(main.instr_panel.h_play_once, 'Enable', 'Off');
        set(main.status_image.title, 'String', {'Disconnected', ' ', ' '});
    end

    function fn_instrument_select(ii)
        %stop continuous acquisition
        set(main.instr_panel.h_play, 'value', 0);
        %call disconnect on previous instrument
        fn_instrument_disconnect;
        %set new current instrument
        setup.current_instr = ii;
        main.instr_panel.fn_set_content(available_instruments(setup.current_instr).instr_info.options_info);
        %enable connect button
        set(main.instr_panel.h_connect, 'Enable', 'On');
        %do not connect until explicitly asked - force connect to off state
        set(main.instr_panel.h_connect, 'Value', 0);
        %make sure current instrument is selected in pulldown (needed for
        %when it is selected from setup file)
        set(main.instr_panel.h_instrument, 'Value', setup.current_instr);
    end

    function fn_play(play_once)
        %get array and material
        data_is_from_file = 0;
        array = fn_get_array;
        [ph_velocity, vel_poly] = fn_get_ph_velocity;
        if (isempty(array) | isempty(ph_velocity)) & setup.current_instr > 1
            set(main.instr_panel.h_play, 'value', 0);
            return;
        end;
        try
            if get(main.instr_panel.h_play, 'value') | play_once
                fps_string = '';
                %play case - send set up
                set(main.status_image.title, 'String', {...
                    ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                    'Sending set up ...', fps_string});
                if setup.current_instr > 1
                    available_instruments(setup.current_instr).fn_send_instr_options(...
                        main.instr_panel.fn_get_data(), length(array.el_xc));
                else
                    %If its the emulator
                    available_instruments(setup.current_instr).fn_send_instr_options(...
                        main.instr_panel.fn_get_data(), 0);
                end
                set(main.status_image.title, 'String', {...
                    ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                    'Sending set up ... done', fps_string});
                if ~play_once
                    set(main.instr_panel.h_play, 'CData', fn_get_cdata_for_named_icon(icons, 'stop'));
                end
            else
                %this is the stop case!
                set(main.status_image.title, 'String', {...
                    ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                    'Stopped', ' '});
                if ~play_once
                    set(main.instr_panel.h_play, 'CData', fn_get_cdata_for_named_icon(icons, 'play'));
                end
                return
            end
            repeat = 1;
            while (get(main.instr_panel.h_play, 'value') & ~play_once) | (repeat & play_once)
                set(main.status_image.title, 'String', {...
                    ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                    'Waiting for data ...', ...
                    fps_string});
                exp_data = available_instruments(setup.current_instr).fn_instr_acquire(main.instr_panel.fn_get_data());
                if ~isempty(exp_data)
                    if ~isempty(time_of_last_frame) &~play_once
                        fps = 1 / toc(time_of_last_frame);
                        if fps > 1;
                            fps_string = sprintf('Frames/second: %.2f ', fps);
                        else
                            fps_string = sprintf('Seconds/frame: %.2f ', 1 / fps);
                        end;
                    else
                        fps_string = '';
                    end;
                    time_of_last_frame = tic;
                    set(main.status_image.title, 'String', {...
                        ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                        'Waiting for data ... done', ...
                        fps_string});
                    if setup.current_instr > 1
                        exp_data.ph_velocity = ph_velocity;
                        if ~isempty(vel_poly)
                            exp_data.vel_poly = vel_poly;
                        end;
                        exp_data.array = array;
                    end
                    fn_update;
                else
                    %if array or material not set, stop play
                    set(main.instr_panel.h_play, 'value', 0);
                end
                if play_once
                    repeat = 0;
                    set(main.status_image.title, 'String', {...
                        ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                        'Stopped', ' '});
                end
            end
        catch
            %just in case window is closing
        end
    end

    function fn_instrument_reset
        set(main.instr_panel.h_play, 'value', 0);%stop play if it was running
        available_instruments(setup.current_instr).fn_instr_reset()
    end

    function fn_material_select(mi)
        %just need to check if last item is selected, as that is Create new
        %material
        if data_is_from_file && ~callbacks_disabled
            b = questdlg('Ignore material in file?','Material','Yes', 'No', 'Yes');
            if strcmp(b, 'No')
                return;
            end
        end
        
        if mi == length(get(main.instr_panel.h_material, 'String'))
            [matl, fname] = fn_input_new_material_details(fullfile(local_folder, config.files.local_brain_path, config.files.materials_path));
            if isempty(matl)
                return;
            end
            jj = length(available_materials) + 1;
            available_materials(jj).material = matl;
            available_materials(jj).name = fname;
            fn_populate_materials_pulldown;
        end
        setup.current_matl= mi;
        callbacks_disabled = 1;
        set(main.instr_panel.h_material, 'value', mi);
        callbacks_disabled = 0;
        if data_is_from_file
            exp_data.ph_velocity = available_materials(setup.current_matl).material.ph_velocity;
            if isfield(available_materials(setup.current_matl).material, 'vel_poly')
                exp_data.vel_poly = available_materials(setup.current_matl).material.vel_poly;
            end
        end
    end

    function fn_array_select(mi)
        %just need to check if last item is selected, as that is Create new
        %material
        if mi == length(get(main.instr_panel.h_array, 'String'))
            [array, fname] = fn_input_new_array_details(fullfile(local_folder, config.files.local_brain_path, config.files.arrays_path));
            if isempty(array)
                return;
            end
            jj = length(available_arrays) + 1;
            available_arrays(jj).array = array;
            available_arrays(jj).name = fname;
            fn_populate_arrays_pulldown;
        end
        setup.current_array = mi;
        callbacks_disabled = 1;
        set(main.instr_panel.h_array, 'value', mi);
        callbacks_disabled = 0;
    end

    function array = fn_get_array
        ai = get(main.instr_panel.h_array, 'value');
        %check it is a real array - first item in list is no selection
        array = [];
        if ai > 1
            array = available_arrays(ai).array;
        else
            if setup.current_instr > 1
                warndlg('Must select array first','Warning')
            end
        end
    end

    function [ph_velocity, vel_poly] = fn_get_ph_velocity
        mi = get(main.instr_panel.h_material, 'value');
        %check it is a real matl - first item in list is no selection and
        %last item is create new material
        ph_velocity = [];
        vel_poly = [];
        if mi > 1 & mi < length(get(main.instr_panel.h_material, 'String'));
            ph_velocity = available_materials(mi).material.ph_velocity;
            if isfield(available_materials(mi).material, 'vel_poly')
                vel_poly = available_materials(mi).material.vel_poly;
            end
        else
            if setup.current_instr > 1
                warndlg('Must select material first','Warning')
            end
        end
    end

    function fn_update
        %this is executed whenever new exp_data is loaded or acquired
        if ~isempty(exp_data)
            set(main.status_image.title, 'String', {...
                ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                'Updating raw data display ...', fps_string});
            %extract amplitude data for FMC display
            n = length(exp_data.array.el_xc);
            
            cdata = ones(n^2,1) * config.general.window_bg_color;
            amp = -ones(n^2, 1);
            %             amp = zeros(n, n);
            ops = main.raw_panel.fn_get_display_options([]);
            if isempty(ops.select)
                min_time = 0;
            else
                min_time = ops.select(1,1);
            end;
            jj = min(find(exp_data.time > min_time));
            if ~isempty(jj)
                amp(sub2ind([n, n], exp_data.rx, exp_data.tx)) = max(abs(exp_data.time_data(jj:end, :)));
                j1 = find(amp > config.main_win.status_panel.max_fract_fsd);
                j2 = find(amp >= config.main_win.status_panel.min_fract_fsd & amp <= config.main_win.status_panel.max_fract_fsd);
                j3 = find(amp < config.main_win.status_panel.min_fract_fsd & amp >= 0);
                for ii = 1:3
                    cdata(j1, ii) = config.main_win.status_panel.too_big_color(ii);
                    cdata(j2, ii) = config.main_win.status_panel.good_color(ii);
                    cdata(j3, ii) = config.main_win.status_panel.too_small_color(ii);
                end
                set(main.status_image.handle, 'CData', reshape(cdata, n, n, 3));
                %                 set(main.status_image.handle, 'CData', amp);
                h = get(main.status_image.handle, 'Parent');
                set(h, 'XLim', [0.5, n+0.5], 'YLim', [0.5, n+0.5], 'XTick', [1, n], 'YTick', [1, n]);
            end
            drawnow;
            
            %plot the raw data
            data.x = exp_data.time;
            ii = find(exp_data.tx == exp_data.rx);
            if length(ii) == 1
                %handle CSM data
                ii = 1:length(exp_data.tx);
            end;
            data.z = exp_data.rx(ii);
            data.f = exp_data.time_data(:, ii)';
            main.raw_panel.fn_update(data);
            
            set(main.status_image.title, 'String', {...
                ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                'Updating raw data display ... done', fps_string});
            
            %loop through all the processes and pump the data out to them
            if isfield(setup, 'gui_process_window')
                for ii = 1:length(setup.gui_process_window)
                    try
                        set(main.status_image.title, 'String', {...
                            ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                            sprintf('Updating image %i of %i ...', ii, length(setup.gui_process_window)), fps_string});
                        drawnow;
                        if ~isempty(setup.gui_process_window(ii).fn_update_data_handle)
                            if recalc_focal_laws
                                [proc_ops, disp_ops] = setup.gui_process_window(ii).fn_get_options_handle([]);
                                proc_ops.options_changed = 1;
                                ops.process_options = proc_ops;
                                setup.gui_process_window(ii).fn_set_options_handle(ops);
                                setup.gui_process_window(ii).fn_update_data_handle(exp_data, 'recalc_and_process');
                                recalc_focal_laws = 0;
                            else
                                setup.gui_process_window(ii).fn_update_data_handle(exp_data, 'process_only');
                            end
                        end
                        set(main.status_image.title, 'String', {...
                            ['Connected to ', available_instruments(setup.current_instr).instr_info.name], ...
                            sprintf('Updating image %i of %i ... done', ii, length(setup.gui_process_window)), fps_string});
                        drawnow;
                    catch
                        %nesc because sometimes windows may be closed
                        %mid-loop
                    end
                end
            end
        else
            data = [];
            main.raw_panel.fn_update(data);
            n = 64;
            cdata = ones(n^2,1) * config.general.window_bg_color;
            set(main.status_image.handle, 'CData', reshape(cdata, n, n, 3));
        end
    end

    function fn_reset
        %this is executed after initial setup whenever new setup is loaded
        
        %delete all child windows
        if isfield(setup, 'gui_process_window')
            try
                for ii = 1:length(setup.gui_process_window)
                    close(setup.gui_process_window(ii).figure_handle);
                end
            catch
            end
        end
        
        %set up figure from scratch
        fn_apply_setup;
    end

    function fn_update_setup_variable
        %ensures that setup is up to date (used prior to saving set up and
        %also when programme exits)
        if isfield(setup, 'gui_process_window')
            for ii = 1:length(setup.gui_process_window)
                [setup.gui_process_window(ii).process_options, setup.gui_process_window(ii).display_options] = setup.gui_process_window(ii).fn_get_options_handle([]);
                setup.gui_process_window(ii).position = get(setup.gui_process_window(ii).figure_handle, 'Position');
            end
        end
        setup.instr_options = main.instr_panel.fn_get_data();
        setup.main_win.non_maximised_position = get(main.handle, 'OuterPosition');
        setup.raw_display_options = main.raw_panel.fn_get_display_options([]);
        if config.general.save_last_exp_data
            setup.last_exp_data = exp_data; %needed because process windows require some of exp_data to open ...
        else
            setup.last_exp_data = [];
        end
    end

    function fn_close_main_window(src, ev) %captures the data from all process windows as they close and saves them
        ready_for_data = 0;
        
        available_instruments(setup.current_instr).fn_instr_disconnect(); %disconnect instrument
        %fn_instr_disconnect(); %this line shows how this was setup before
        main_window_closing = 1;
        %update setup variable
        fn_update_setup_variable;
        %close all the child windows
        if isfield(setup, 'gui_process_window')
            for ii = 1:length(setup.gui_process_window)
                close(setup.gui_process_window(ii).figure_handle);
            end
        end
        %save the setup as "last setup.ndt"
        if config.general.save_last_setup
            save(fullfile(local_folder, config.files.local_brain_path, config.files.setups_path, config.files.last_setup_file), 'setup');
        end
        %finally change back to directory from which gui_main_window was called
        if ~isdeployed
            cd(current_dir_when_brain_started);
        end
    end

    function fn_instr_options_changed(ops)
        recalc_focal_laws = 1;
        %         keyboard
        array = fn_get_array;
        if isempty(array)
            no_els = 0;
        else
            no_els = length(array.el_xc);
        end;
        if get(main.instr_panel.h_play, 'value')
            set(main.instr_panel.h_play, 'value', 0);
            was_playing = 1;
        else
            was_playing = 0;
        end
        available_instruments(setup.current_instr).fn_send_instr_options(ops, no_els);
        if was_playing
            set(main.instr_panel.h_play, 'value', 1);
        end
    end
end

function fname = fn_save_file(exp_data, setup, folder)
filter{1,1} = '*.mat'; filter{1,2} = 'Array data file (*.mat)';
filter{2,1} = '*.ndt'; filter{2,2} = 'Setup file (*.ndt)';
[fname, pathname, filterindex] = uiputfile(filter, 'Save', [folder, filesep]);
if ~ischar(fname)
    %nothing selected
    fname = [];
    return;
end
switch filterindex
    case 1
        %prevent save without array or material data
        if ~isfield(exp_data, 'array')
            warndlg('Select array first','Warning')
            return;
        end;
        if ~isfield(exp_data, 'ph_velocity')
            warndlg('Select material first','Warning')
            return;
        end;
        save([pathname, fname], 'exp_data');
    case 2
        save([pathname, fname], 'setup');
end
fname =fullfile(pathname, fname);
end

function [exp_data, setup, exp_data_fname] = fn_load_file(folder, array_folder)
exp_data_fname = [];%only set if exp_data file is loaded
filter{1,1} = '*.mat'; filter{1,2} = 'Array data file (*.mat)';
filter{2,1} = '*.ndt'; filter{2,2} = 'Setup file (*.ndt)';
filter{3,1} = '*.png'; filter{3,2} = 'Diagnostic sonar file (*.png)';
filter{4,1} = '*.txt'; filter{4,2} = 'M2M CIVA file (*.txt)';

[fname, pathname, filterindex] = uigetfile(filter, 'Load', folder);

if ~ischar(fname)
    %nothing selected
    exp_data = [];
    setup = [];
    exp_data_fname = [];
    return;
end

if strcmp(fname(end-2:end),'png')==1
    exp_data=fn_ds_convert([pathname, fname(1:end-4)]);
elseif strcmp(fname(end-2:end),'txt')==1
    exp_data=fn_m2m_convert([pathname, fname(1:end-4)]);
else
    load([pathname, fname], '-mat');
     
    if exist('Trans')
       [exp_data]=fn_verasonics_convert(Trans, Receive, RcvData);
    end
end


if exist('exp_data') == 1
    setup = [];
    success = 0;
    while ~success
        success = 1;
        if isfield(exp_data, 'array')
            res = questdlg('Which array information would you like to use?', '','From current file', 'From different file','Create new array', 'From current file');
        else
            res = questdlg('Which array information would you like to use?', '','From different file','Create new array', 'From different file');
        end
        switch res
            case 'From different file'
                [fname, pathname, filterindex] = uigetfile({'*.mat', 'Array file (*.mat)'}, 'Load', array_folder);
                tmp = load([pathname, fname], '-mat');
                if isfield(tmp, 'array')
                    exp_data.array = tmp.array;
                else
                    success = 0;
                end
            case 'Create new array'
                [exp_data.array, fname] = fn_input_new_array_details(array_folder);
        end
    end
    exp_data_fname = [pathname, fname];
    return;
end
if exist('setup') == 1
    exp_data = [];
    return;
end
fname =fullfile(pathname, fname);
end
