function config = fn_get_config;
%Contains all configuration variables for brain

%Default setup
config.default_setup.main_win.non_maximised_position = [0.1, 0.1, 0.8, 0.8];
config.default_setup.current_instr = 1;
config.default_setup.current_matl = 1;
config.default_setup.current_array = 1;
config.default_setup.last_exp_data = [];

%General system info
config.general.screen_size_pixels = get(0, 'ScreenSize');
config.general.button_height_pixels = 24;
config.general.status_height_pixels = 48;
config.general.slider_width_pixels = 24;
config.general.window_bg_color = [0.8, 0.8, 0.8];
config.general.save_last_exp_data = 0;
config.general.save_last_setup = 0;
config.general.load_last_setup = 0;

%General color scheme and geometry defaults
config.array_el_edge_color = 'r';
config.array_el_patch_color = 'r';
config.array_ellipt_el_pts = 13;
config.default_geom_linestyle = '-';

%Files and file locations
config.files.local_brain_path = 'Brain (local data)';
config.files.imaging_path = 'Imaging';
config.files.instruments_path = 'Instruments';
config.files.arrays_path = 'Arrays';
config.files.materials_path = 'Materials';
config.files.analysis_path = 'Analysis';
config.files.setups_path = 'Setups';

config.files.last_setup_file = 'last setup.ndt';

config.files.icon_file = ['Icons', filesep, 'Icons3.mat'];
config.files.original_matlab_icon_file = ['Icons', filesep, 'original-matlab-icons.mat'];

%Main window details - title and version
config.main_win.title = 'BRAIN';
config.main_win.version = 1.9;
config.main_win.non_maximised_position = [0.1, 0.1, 0.8, 0.8];
config.main_win.gpu = 0;

%Main window details - frames per second
config.main_win.min_fps_to_show = 0.1;

%Main window details - raw data display
config.main_win.raw_data_panel.position = [0.25, 0, 0.75, 1];
config.main_win.raw_data_panel.border_type = 'etchedin';
config.main_win.raw_data_panel.time_axis_scalefactor = 1e6;
config.main_win.raw_data_panel.vert_axis_scalefactor = 1;
config.main_win.raw_data_panel.time_axis_label = 'Time (us)';
config.main_win.raw_data_panel.vert_axis_label = 'Receiver number';

%Main window details - status panel
config.main_win.status_panel.border_type = 'etchedin';
config.main_win.status_panel.good_color = [0,1,0];
config.main_win.status_panel.too_big_color = [1,0,0];
config.main_win.status_panel.too_small_color = [1,1,0];
config.main_win.status_panel.position = [0, 0, 0.25, 0.4];
config.main_win.status_panel.axes_position = [0, 0.2, 1, 0.8];
config.main_win.status_panel.title_position = [0.1, 0, 0.9, 0.2];
config.main_win.status_panel.font_size = 8;

%Main window details - status panel - fractions of full scale deflection
%that count as too big or too small
config.main_win.status_panel.max_fract_fsd = 0.8;
config.main_win.status_panel.min_fract_fsd = 0.2;

%Main window details - instrument panel
config.main_win.instr_panel.position = [0, 0.4, 0.25, 0.6];
config.main_win.instr_panel.border_type = 'etchedin';

%Main window details - delay when connecting to instrument
config.main_win_instr_connect_delay = 0.2;

%Plot panel (2D)
config.plot_panel_2d.border_type = 'etchedin';
config.plot_panel_2d.side_graph_fraction = 0.2;
config.plot_panel_2d.max_db_range = 100;
config.plot_panel_2d.status_panel.border_type = 'etchedin';
config.plot_panel_2d.control_panel.border_type = 'etchedin';
config.plot_panel_2d.graph_panel.border_type = 'etchedin';

%Plot panel (3D)
config.plot_panel_3d.border_type = 'etchedin';
config.plot_panel_3d.status_panel.border_type = 'etchedin';
config.plot_panel_3d.control_panel.border_type = 'etchedin';
config.plot_panel_3d.graph_panel.border_type = 'etchedin';
config.plot_panel_3d.plot_2d.border_type = 'none';
config.plot_panel_3d.plot_3d.border_type = 'none';
config.plot_panel_3d.isosurf_alpha = 1;
config.plot_panel_3d.plane_color = 'y';
config.plot_panel_3d.plane_alpha = 0.25;
config.plot_panel_3d.active_color = 'r';
config.plot_panel_3d.selection_alpha = 0.25;
config.plot_panel_3d.selection_color = 'm';

%New materials window
config.new_matl_win.pixel_size = [300, 200];

%New array window
config.new_array_win.pixel_size = [600, 200];
config.new_array_win.table_frac = 0.25;
config.new_array_win.default_manufacturer = '';
config.new_array_win.default_cent_freq = 1e6;
config.new_array_win.default_pitch = 1e-3;
config.new_array_win.default_separation = 0.1e-3;
config.new_array_win.default_length = 10e-3;
config.new_array_win.default_no_els = 64;

return