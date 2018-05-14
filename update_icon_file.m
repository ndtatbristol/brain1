%refresh main icons file, using original matlab icons and all other icon
%files in that directory
fn_clear;
config = fn_get_config;
[pt, nm, ext] = fileparts(config.files.icon_file);
icons = fn_get_icons_from_file(config.files.original_matlab_icon_file, pt, config.general.window_bg_color)
save(config.files.icon_file, 'icons');