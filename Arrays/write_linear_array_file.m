%create linear array data file
folder = 'N:\ndt-library\arrays\Bristol arrays\';

%for twin arrays only enter details of one array!
no_els = 64;
el_pitch = 0.5e-3;
el_length = 15e-3;
el_sep = 0.15e-3;
twin_array = 0;
centre_freq = 2.5e6;
manufacturer = 'Imasonic';
%-------------------------------------------------------------------------%
el_width = el_pitch - el_sep;

if twin_array
    tmp = '2x';
else
    tmp = '';
end;
fname = sprintf([manufacturer, ' 1D ', tmp, '%iels %.2fMHz %.2fmm pitch.mat'], no_els, centre_freq / 1e6, el_pitch * 1e3);

array.el_xc = [1:no_els] * el_pitch;
array.el_xc = array.el_xc - mean(array.el_xc);
array.el_yc = zeros(size(array.el_xc));
array.el_zc = zeros(size(array.el_xc));
array.el_x1 = array.el_xc + el_width / 2;
array.el_y1 = zeros(size(array.el_xc));
array.el_z1 = zeros(size(array.el_xc));
array.el_x2 = array.el_xc;
array.el_y2 = array.el_yc + el_length / 2;
array.el_z2 = zeros(size(array.el_xc));

%uncomment following for twin arrays (only define first one!)
if twin_array
    array.el_xc = [array.el_xc, array.el_xc];
    array.el_yc = [array.el_yc, array.el_yc];
    array.el_zc = [array.el_zc, array.el_zc];
    array.el_x1 = [array.el_x1, array.el_x1];
    array.el_y1 = [array.el_y1, array.el_y1];
    array.el_z1 = [array.el_z1, array.el_z1];
    array.el_x2 = [array.el_x2, array.el_x2];
    array.el_y2 = [array.el_y2, array.el_y2];
    array.el_z2 = [array.el_z2, array.el_z2];
end

save([folder, fname], 'array');
disp(fname);