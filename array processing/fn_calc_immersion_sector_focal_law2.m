function [focal_law, surface] = fn_calc_immersion_sector_focal_law2(exp_data, mesh, orig_surface, immersion_options, varargin)
if nargin < 5
    use_gpu_if_available = 1;
else
    use_gpu_if_available = varargin{1};
end
default_immersion_options.centre_freq = exp_data.array.centre_freq;
default_immersion_options.couplant_velocity = 1480;
default_immersion_options.extrapolate_surface = 0;
default_immersion_options.interp_pts_per_sample_wavelength = 0.5; %set to inf to get benchmark result
default_immersion_options.surface_pts_per_sample_wavelength = 5;

%This is basically a modified version of fn_calc_immersion_tfm_focal_law2
%that goes backwards by not focusing in the sample! Ray path from image
%point to centre of array is computed and then delay laws for other
%elements are calculated assuming planewave is sent at that angle.

immersion_options = fn_set_default_fields(immersion_options, default_immersion_options);
sample_wavelength = exp_data.ph_velocity / immersion_options.centre_freq;
surface = orig_surface;

%upsample surface and extrapolate if nesc
if immersion_options.extrapolate_surface
    min_x = min([mesh.x(1,:)'; surface.x(:)]);
    max_x = max([mesh.x(1,:)'; surface.x(:)]);
else
    min_x = min(surface.x(:));
    max_x = max(surface.x(:));
end
tmp_x = linspace(min_x, max_x, max([ceil((max_x - min_x) / sample_wavelength * immersion_options.surface_pts_per_sample_wavelength), 2]));
surface.z = interp1(surface.x, surface.z, tmp_x, 'spline', 'extrap');
surface.x = tmp_x;

%interpolate specified surface onto grid x values - needs to be generalised
%for 2D case to identify valid points
tmp_surf_z = interp1(surface.x(:), surface.z(:), mesh.x(1,:), 'linear', 'extrap');
mesh.valid_pts = mesh.z > ones(size(mesh.z, 1),1) * tmp_surf_z;

%partition mesh - mesh1 is completely above surface and not processed at
%all; mesh2 is the surface region; mesh3 is completely below surface
zi_12 = max([find(mesh.z(:,1) < min(surface.z)); 2]); %last index of couplant-only region

if zi_12 == size(mesh.z, 1)
    %case where entire mesh is in couplant so no calculation required!
    focal_law.lookup_time = zeros(size(mesh.x,1), size(mesh.x,2), length(exp_data.array.el_xc));
    focal_law.lookup_amp = zeros(size(mesh.x,1), size(mesh.x,2), length(exp_data.array.el_xc));
    focal_law.lookup_ind = zeros(size(mesh.x,1), size(mesh.x,2), length(exp_data.array.el_xc));
    focal_law.tt_weight = ones(1, length(exp_data.tx));
    return
end

zi_23 = min(find(mesh.z(:,1) > (max(surface.z) + 2 * sample_wavelength))); %first index of sample-only region

%the actual focal law calculations
%first - at points in the body of sample using interpolation
tmp_mesh.x = mesh.x(zi_23:end,:);
tmp_mesh.z = mesh.z(zi_23:end,:);
[lookup_time3, lookup_amp3] = fn_all_points_method2(tmp_mesh, exp_data.array, surface, exp_data.ph_velocity, sample_wavelength, immersion_options);
%second - at points in the surface region of sample without interpolation
tmp_mesh.x = mesh.x(zi_12: zi_23 - 1,:);
tmp_mesh.z = mesh.z(zi_12: zi_23 - 1,:);
tmp_mesh.valid_pts = mesh.valid_pts(zi_12 + 1: zi_23 - 1,:);
immersion_options.interp_pts_per_sample_wavelength = inf; %force all point calc in this region near surface
[lookup_time2, lookup_amp2] = fn_all_points_method2(tmp_mesh, exp_data.array, surface, exp_data.ph_velocity, sample_wavelength, immersion_options);
%combine them into the final focal law

focal_law.lookup_time = [zeros(size(mesh.x,1) - size(lookup_time2, 1) - size(lookup_time3, 1), size(mesh.x, 2), length(exp_data.array.el_xc)); lookup_time2; lookup_time3];
focal_law.lookup_amp = [zeros(size(mesh.x,1) - size(lookup_time2, 1) - size(lookup_time3, 1), size(mesh.x, 2), length(exp_data.array.el_xc)); lookup_amp2; lookup_amp3];

%zero the amplitudes for all points above surface
for ii = 1:size(focal_law.lookup_time, 3)
    focal_law.lookup_amp(:, :, ii) = focal_law.lookup_amp(:, :, ii) .* mesh.valid_pts;
end

%do the lookup ind for nearest neighbour interpolation
dt = exp_data.time(2) - exp_data.time(1);
t0 = exp_data.time(1);
focal_law.lookup_ind = round((focal_law.lookup_time - t0 / 2) / dt);
focal_law.interpolation_method = 'nearest';

%finally the weightings to account for HMC data
focal_law.tt_weight = ones(1, length(exp_data.tx));
for ii = 1:length(focal_law.tt_weight)
    %check if other combo is present
    tx = exp_data.tx(ii);
    rx = exp_data.rx(ii);
    if tx == rx
        continue;
    end
    if isempty(find(exp_data.tx == rx & exp_data.rx == tx))
        focal_law.tt_weight(ii) = 2;
    end
end
end

%--------------------------------------------------------------------------
function [lookup_time, lookup_amp] = fn_all_points_method2(mesh, array, surface, sample_velocity, sample_wavelength, immersion_options)
if size(mesh.x,1) < 2 || size(mesh.x,2) < 2
    lookup_time = [];
    lookup_amp = [];
    return
end
%number crunching method
lookup_amp = zeros(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
lookup_time = zeros(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));

%downsample mesh if required
pixel_size = min([mesh.x(1,2) - mesh.x(1,1), mesh.z(2,1) - mesh.z(1,1)]);
coarse_pixel_size = sample_wavelength / immersion_options.interp_pts_per_sample_wavelength;
if coarse_pixel_size > pixel_size
    use_coarse_mesh = 1;
else
    use_coarse_mesh = 0;
end

if use_coarse_mesh
    coarse_mesh.x = linspace(min(min(mesh.x)), max(max(mesh.x)), ceil((max(max(mesh.x)) - min(min(mesh.x))) / coarse_pixel_size));
    coarse_mesh.z = linspace(min(min(mesh.z)), max(max(mesh.z)), ceil((max(max(mesh.z)) - min(min(mesh.z))) / coarse_pixel_size));
    [coarse_mesh.x, coarse_mesh.z] = meshgrid(coarse_mesh.x, coarse_mesh.z);
    p = [coarse_mesh.x(:), coarse_mesh.z(:)];
else
    p = [mesh.x(:), mesh.z(:)];
end

%matrix of all surface points
s = [surface.x(:), surface.z(:)]';

%matrix of times from every image point to every surface point
sample_time = zeros(size(p,1), size(s,2));
for ii = 1:size(p,2)
    sample_time = sample_time + (repmat(p(:,ii), [1, size(s, 2)]) - repmat(s(ii,:), [size(p, 1), 1])) .^ 2;
end
sample_time = sqrt(sample_time) / sample_velocity;

%matrix of element positions and normal directions
e = [array.el_xc(:), array.el_zc(:)];
e_theta = atan2(-array.el_xc(:) + array.el_x1(:), array.el_zc(:) - array.el_z1(:));
%array centre position
e_centre = mean(e);
e_theta_centre = mean(e_theta);

%array normal direction
n = fn_return_array_normal_vector(array);
%matrices of times and angles from every element to centre of array
dd = zeros(1, size(s,2), size(s,1));

for ii = 1:size(s,1)
    dd(:,:,ii) = repmat(s(ii,:), [size(e_centre, 1), 1]) - repmat(e_centre(:,ii), [1, size(s, 2)]);
end
dxy = sqrt(sum(dd(:,:,1:end-1) .^ 2, 3));
dz = dd(:,:,end);
dd_dot_n = sum(dd .* repmat(reshape([n(1:size(s,1)-1), n(end)], [1,1,size(s,1)]), [1, size(s,2), 1]), 3);
d = sqrt(sum(dd .^ 2, 3));
couplant_angs = acos(dd_dot_n ./ d);

couplant_angs = atan2(repmat(s(2,:), [size(e_centre, 1), 1]) - repmat(e_centre(:,2), [1, size(s, 2)]), ...
    repmat(s(1,:), [size(e_centre, 1), 1]) - repmat(e_centre(:,1), [1, size(s, 2)])) - repmat(e_theta_centre, [1, size(s,2)]);

couplant_time = d / immersion_options.couplant_velocity;

%loop over elements and find min time for each, checking that ends of
%surface are not used
total_time = sample_time + repmat(couplant_time(1, :), [size(p, 1),1]);
[min_time, jj] = min(total_time, [], 2);
amp = double((jj > 1) & (jj < length(surface.x)));
if immersion_options.angle_limit_on && immersion_options.max_angle_in_couplant > 0
    amp = amp .* (abs(couplant_angs(1, jj)) <= immersion_options.max_angle_in_couplant)';
    %note there is no distance compensation for fixed angle here as it
    %is not clear how to do it (cf. contact TFM)
end
for ii = 1:length(array.el_xc)
    delta = sqrt(sum((e(ii,:) - e(1,:)) .^ 2))-sqrt(sum((e_centre - e(1,:)) .^ 2));
    tmp_min_time = min_time + delta * tan(couplant_angs(1, jj)).' / immersion_options.couplant_velocity;
    if use_coarse_mesh
        coarse_amp = reshape(amp, size(coarse_mesh.x));
        coarse_time = reshape(tmp_min_time, size(coarse_mesh.x));
        lookup_amp(:,:,ii) = interp2(coarse_mesh.x, coarse_mesh.z, coarse_amp, mesh.x, mesh.z, 'linear');
        lookup_time(:,:,ii) = interp2(coarse_mesh.x, coarse_mesh.z, coarse_time, mesh.x, mesh.z, 'linear');
    else
        lookup_amp(:,:,ii) = reshape(amp, size(mesh.x));
        lookup_time(:,:,ii) = reshape(tmp_min_time, size(mesh.x));
    end
end
end
