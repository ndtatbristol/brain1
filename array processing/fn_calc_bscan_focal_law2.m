function focal_law = fn_calc_bscan_focal_law2(exp_data, mesh, varargin)
%SUMMARY
%   Computes the focal law for Bscan for use with fn_fast_DAS2. This is the
%   gneral 2D or 3D version depending on fields of mesh (x and z / x,y and z).
%INPUTS
%   exp_data - a representative set of experimental data, from which the
%   array geometry, time axis etc will be derived
%   mesh.x, [mesh.y,] mesh.z - matrices of image point coordinates (typically created using
%   meshgrid command)
%   options [] - optional structured variable of special options, such as
%   aperture_size_els or aperture_size_metres -  aperture_size_els is
%   assumed to be relative to the minimum inter-element spacing in any
%   array
%OUTPUTS
%   focal_law - all that's necessary to use with fn_fast_DAS2 function
if nargin < 3
    options = [];
else
    options = varargin{1};
end;
default_options.aperture_size_els = 8;
default_options.beam_angle = 0;

options = fn_set_default_fields(options, default_options);

if isfield(options, 'aperture_size_metres')
    aperture_size = options.aperture_size_metres;
else
    %calc min interelement spacing
    dx = exp_data.array.el_xc(:) * ones(1, length(exp_data.array.el_xc));
    dy = exp_data.array.el_yc(:) * ones(1, length(exp_data.array.el_yc));
    dz = exp_data.array.el_zc(:) * ones(1, length(exp_data.array.el_zc));
    dx = dx - dx';
    dy = dy - dy';
    dz = dz - dz';
    dx = dx + diag(ones(size(dx, 1), 1) * inf);
    dy = dy + diag(ones(size(dy, 1), 1) * inf);
    dz = dz + diag(ones(size(dz, 1), 1) * inf);
    r = sqrt(dx .^ 2 + dy .^ 2 + dz .^ 2);
    min_el_sep = min(min(r));
    aperture_size = options.aperture_size_els * min_el_sep;
end
if isfield(mesh, 'y')
    n = 3;
else
    n = 2;
end
if n == 2
    dist = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    for ii = 1:length(exp_data.array.el_xc)
        dist(:, :, ii) = mesh.z - exp_data.array.el_zc(ii);
    end;
else
    dist = zeros(size(mesh.x, 1), size(mesh.x, 2),  size(mesh.x, 3), length(exp_data.array.el_xc));
    for ii = 1:length(exp_data.array.el_xc)
        dist(:, :, :, ii) = mesh.z - exp_data.array.el_zc(ii);
    end;
end

dt = exp_data.time(2) - exp_data.time(1);
t0 = exp_data.time(1);

v = exp_data.ph_velocity;

focal_law.lookup_time = dist ./ v;
focal_law.lookup_ind = round((focal_law.lookup_time - t0 / 2) / dt);
focal_law.lookup_amp = zeros(size(focal_law.lookup_ind));
for ii = 1:length(exp_data.array.el_xc)
    if n == 2
        focal_law.lookup_amp(:, :, ii) = abs(mesh.x - exp_data.array.el_xc(ii)) <= (aperture_size / 2);
    end
    if n == 3
        focal_law.lookup_amp(:, :, :, ii) = sqrt(...
            (mesh.x - exp_data.array.el_xc(ii)) .^ 2 + ...
            (mesh.y - exp_data.array.el_yc(ii)) .^ 2) ...
            <= (aperture_size / 2);
    end
end;

focal_law.filter_on = 0;
focal_law.hilbert_on = 1;

%add time-trace weighting vector to account for missing pitch-catch data
%if half matrix capture is used
focal_law.tt_weight = ones(1, length(exp_data.tx));
for ii = 1:length(focal_law.tt_weight)
    %check if other combo is present
    tx = exp_data.tx(ii);
    rx = exp_data.rx(ii);
    if tx == rx
        continue;
    end;
    if isempty(find(exp_data.tx == rx & exp_data.rx == tx))
        focal_law.tt_weight(ii) = 2;
    end;
end;

focal_law.tt_ind = fn_optimise_focal_law2(focal_law, exp_data.tx, exp_data.rx);
focal_law.interpolation_method = 'nearest';
return;