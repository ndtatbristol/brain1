function array_geom = fn_get_array_geom_for_plots(array)
%function returns coordinate arguments (x, y and z) for subsequent call of patch or line command

config = fn_get_config;

dx = array.el_x1(:)' - array.el_xc(:)';
if all(dx == 0)
    dx = array.el_x2(:)' - array.el_xc(:)';
end    
dy = array.el_y2(:)' - array.el_yc(:)';
if all(dy == 0)
    dy = array.el_y1(:)' - array.el_yc(:)';
end
xc = array.el_xc(:)';
yc = array.el_yc(:)';
zc = array.el_zc(:)';

if isfield(array, 'el_type') && strcmp(array.el_type, 'elliptical')
    a = linspace(0, 2 * pi, config.array_ellipt_el_pts)';
    x = ones(size(a)) * xc + cos(a) * dx
    y = ones(size(a)) * yc + sin(a) * dy
else
    x = [xc - dx; xc + dx; xc + dx; xc - dx; xc - dx];
    y = [yc - dy; yc - dy; yc + dy; yc + dy; yc - dy];
end
z = ones(size(x,1), 1) * zc;
array_geom.x = x;
array_geom.y = y;
array_geom.z = z;
end