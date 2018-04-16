function focal_law = fn_calc_csm_with_ddf_focal_law2(exp_data, mesh, varargin)
%SUMMARY
%   Computes the focal law for common source method (CSM) and dynamic depth
%   focusing on reception for use with fn_fast_DAS2.
%   This is the general 2D or 3D version depending on fields in mesh (x and
%   z or x, y, and z)
%INPUTS
%   exp_data - a representative set of experimental data, from which the
%   array geometry, time axis etc will be derived
%   x, z - matrices of image point coordinates (typically created using
%   meshgrid command)
%   options [] - optional structured variable of special options, such as
%   limited aperture angle and forcing use of angle dependent velocity
%   (requires exp_data.vel_poly.mu and exp_data.vel_poly.p to exist).
%OUTPUTS
%   focal_law - all that's necessary to use with fn_fast_DAS2 function

if nargin < 3
    options = [];
else
    options = varargin{1};
end;
default_options.angle_dep_vel = 0;
default_options.angle_limit = 0;

options = fn_set_default_fields(options, default_options);

if isfield(mesh, 'y')
    n = 3;
else
    n = 2;
end

if n == 2
    dist_tx = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    dist_rx = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    ang = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    for ii = 1:length(exp_data.array.el_xc)
        dist_tx(:, :, ii) = abs(mesh.z - exp_data.array.el_zc(ii));
        dist_rx(:, :, ii) = sqrt((mesh.x - exp_data.array.el_xc(ii)) .^ 2 + (mesh.z - exp_data.array.el_zc(ii)) .^ 2);
        ang(:, :, ii) = atan2(mesh.x - exp_data.array.el_xc(ii), mesh.z - exp_data.array.el_zc(ii));
    end;
else
    dist_tx = zeros(size(mesh.x, 1), size(mesh.x, 2), size(mesh.x, 3), length(exp_data.array.el_xc));
    dist_rx = zeros(size(mesh.x, 1), size(mesh.x, 2), size(mesh.x, 3), length(exp_data.array.el_xc));
    ang = zeros(size(mesh.x, 1), size(mesh.x, 2), size(mesh.x, 3), length(exp_data.array.el_xc));
    for ii = 1:length(exp_data.array.el_xc)
        dist_tx(:, :, :, ii) = abs(mesh.z - exp_data.array.el_zc(ii));
        dist_rx(:, :, :, ii) = sqrt((mesh.x - exp_data.array.el_xc(ii)) .^ 2 + (mesh.y - exp_data.array.el_yc(ii)) .^ 2 + (mesh.z - exp_data.array.el_zc(ii)) .^ 2);
        ang(:, :, :, ii) = atan2(sqrt((mesh.x - exp_data.array.el_xc(ii)) .^ 2 + (mesh.y - exp_data.array.el_yc(ii)) .^ 2), mesh.z - exp_data.array.el_zc(ii));
    end;
end
dt = exp_data.time(2) - exp_data.time(1);
t0 = exp_data.time(1);

if options.angle_dep_vel & isfield(exp_data, 'vel_poly')
    v_tx = polyval(exp_data.vel_poly.p, 0, [], exp_data.vel_poly.mu);
    v_rx = polyval(exp_data.vel_poly.p, abs(ang), [], exp_data.vel_poly.mu);
else
    v_tx = exp_data.ph_velocity;
    v_rx = exp_data.ph_velocity;
end;

focal_law.lookup_time_tx = dist_tx ./ v_tx;
focal_law.lookup_time_rx = dist_rx ./ v_rx;
focal_law.lookup_ind_tx = round((focal_law.lookup_time_tx - t0 / 2) / dt);
focal_law.lookup_ind_rx = round((focal_law.lookup_time_rx - t0 / 2) / dt);
if n == 2
    focal_law.lookup_amp_tx = ones(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
else
    focal_law.lookup_amp_tx = ones(size(mesh.x, 1), size(mesh.x, 2), size(mesh.x, 3), length(exp_data.array.el_xc));
end
if options.angle_limit
    if n == 2
        focal_law.lookup_amp_rx = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    else
        focal_law.lookup_amp_rx = zeros(size(mesh.x, 1), size(mesh.x, 2), size(mesh.x, 3), length(exp_data.array.el_xc));
    end
    for ii = 1:length(exp_data.array.el_xc)
        if n == 2
            focal_law.lookup_amp_rx(:, :, ii) = (abs(ang(:, :, ii)) < options.angle_limit) ./ mesh.z .* sqrt(dist_rx(:,:,ii));
        else
            focal_law.lookup_amp_rx(:, :, :, ii) = (abs(ang(:, :, :, ii)) < options.angle_limit) ./ mesh.z .* sqrt(dist_rx(:,:,:,ii));
        end
    end;
    focal_law.amp_on = 1;
else
    if n == 2
        focal_law.lookup_amp_rx = ones(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    else
        focal_law.lookup_amp_rx = ones(size(mesh.x, 1), size(mesh.x, 2), size(mesh.x, 3), length(exp_data.array.el_xc));
    end
    focal_law.amp_on = 0;
end;
focal_law.filter_on = 0;
focal_law.hilbert_on = 1;

%add time-trace weighting vector to account for missing pitch-catch data
%if half matrix capture is used
focal_law.tt_weight = ones(1, length(exp_data.tx));
focal_law.hmc_data = 0;

for ii = 1:length(focal_law.tt_weight)
    %check if other combo is present
    tx = exp_data.tx(ii);
    rx = exp_data.rx(ii);
    if tx == rx
        continue;
    end;
    if isempty(find(exp_data.tx == rx & exp_data.rx == tx))
        focal_law.tt_weight(ii) = 2;
        focal_law.hmc_data = 1;
    end;
end;
focal_law.new_focal_law = 1;
focal_law.interpolation_method = 'nearest';
focal_law.range_check = 1;
focal_law.tt_ind = find(focal_law.tt_weight);
return;