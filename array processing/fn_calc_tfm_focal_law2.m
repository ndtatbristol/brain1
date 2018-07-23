function focal_law = fn_calc_tfm_focal_law2(exp_data, mesh, varargin)
%USAGE
%   focal_law = fn_calc_focal_law2(exp_data, mesh, [options])
%SUMMARY
%   Computes the focal law for TFM for use with fn_fast_DAS2. This is the
%   gneral 2D or 3D version depending on fields of mesh (x and z; x,y and z)
%INPUTS
%   exp_data - a representative set of experimental data, from which the
%   array geometry, time axis etc will be derived
%   mesh.x, [mesh.y,] mesh.z - matrices of image point coordinates (typically created using
%   meshgrid command)
%   options [] - optional structured variable of special options, such as
%   limited aperture angle and forcing use of angle dependent velocity
%   (requires exp_data.vel_poly.mu and exp_data.vel_poly.p to exist).
%OUTPUTS
%   focal_law - all that's necessary to use with fn_fast_DAS function
%NOTES
%   if options.angle_limit is non-zero then aperture angle at every image
%   point (i.e. f-number) is limited

if nargin < 3
    options = [];
else
    options = varargin{1};
end;
default_options.angle_dep_vel = 0;
default_options.angle_limit = 0;
default_options.interpolation_method = 'linear';
default_options.load_kernel = 1;

options = fn_set_default_fields(options, default_options);

if isfield(mesh, 'y')
    n = 3;
else
    n = 2;
end
if n == 2
    dist = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    ang = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    for ii = 1:length(exp_data.array.el_xc)
        dist(:, :, ii) = sqrt((mesh.x - exp_data.array.el_xc(ii)) .^ 2 + (mesh.z - exp_data.array.el_zc(ii)) .^ 2);
        ang(:, :, ii) = atan2(mesh.x - exp_data.array.el_xc(ii), mesh.z - exp_data.array.el_zc(ii));
    end;
else
    dist = zeros(size(mesh.x, 1), size(mesh.x, 2),  size(mesh.x, 3), length(exp_data.array.el_xc));
    ang = zeros(size(mesh.x, 1), size(mesh.x, 2),  size(mesh.x, 3), length(exp_data.array.el_xc));
    for ii = 1:length(exp_data.array.el_xc)
        dist(:, :, :, ii) = sqrt((mesh.x - exp_data.array.el_xc(ii)) .^ 2 + (mesh.y - exp_data.array.el_yc(ii)) .^ 2 + (mesh.z - exp_data.array.el_zc(ii)) .^ 2);
        ang(:, :, :, ii) = atan2(sqrt((mesh.x - exp_data.array.el_xc(ii)) .^ 2 + (mesh.y - exp_data.array.el_yc(ii)) .^ 2), mesh.z - exp_data.array.el_zc(ii));
    end;
end

dt = exp_data.time(2) - exp_data.time(1);
t0 = exp_data.time(1);

if options.angle_dep_vel & isfield(exp_data, 'vel_poly')
    v = polyval(exp_data.vel_poly.p, abs(ang), [], exp_data.vel_poly.mu);
else
    v = exp_data.ph_velocity;
end;

focal_law.lookup_time = dist ./ v;
focal_law.lookup_ind = round((focal_law.lookup_time - t0 / 2) / dt);
if options.angle_limit
    if n == 2
        focal_law.lookup_amp = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
        for ii = 1:length(exp_data.array.el_xc)
            focal_law.lookup_amp(:, :, ii) = (abs(ang(:, :, ii)) < options.angle_limit) ./ (dist(:, :, ii) .* cos(ang(:, :, ii))) .* sqrt(dist(:,:,ii));
        end;
    else
        focal_law.lookup_amp = zeros(size(mesh.x, 1), size(mesh.x, 2), size(mesh.x, 3), length(exp_data.array.el_xc));
        for ii = 1:length(exp_data.array.el_xc)
            focal_law.lookup_amp(:, :, :, ii) = (abs(ang(:, :, :, ii)) < options.angle_limit) ./ (dist(:, :, :, ii) .* cos(ang(:, :, :, ii))) .* sqrt(dist(:,:,:,ii));
        end;
    end
else
    if n == 2
        focal_law.lookup_amp = ones(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    else
        focal_law.lookup_amp = ones(size(mesh.x, 1), size(mesh.x, 2), size(mesh.x, 3), length(exp_data.array.el_xc));
    end
end;
focal_law.lookup_amp(find(isnan(focal_law.lookup_amp))) = 0;
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
focal_law.interpolation_method = options.interpolation_method;

focal_law.tt_ind = fn_optimise_focal_law2(focal_law, exp_data.tx, exp_data.rx);

if options.load_kernel
    [focal_law]=fn_load_kernel(focal_law);
end
return;