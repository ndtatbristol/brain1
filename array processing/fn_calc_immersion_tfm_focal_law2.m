function [focal_law, surface] = fn_calc_immersion_tfm_focal_law2(exp_data, mesh, orig_surface, immersion_options, varargin)
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

%matrix of element positions
e = [array.el_xc(:), array.el_zc(:)];
%array normal direction
n = fn_return_array_normal_vector(array);
%matrices of times and angles from every element to every surface point
dd = zeros(length(array.el_xc), size(s,2), size(s,1));

for ii = 1:size(s,1)
    dd(:,:,ii) = repmat(s(ii,:), [size(e, 1), 1]) - repmat(e(:,ii), [1, size(s, 2)]);
end
dxy = sqrt(sum(dd(:,:,1:end-1) .^ 2, 3));
dz = dd(:,:,end);
dd_dot_n = sum(dd .* repmat(reshape([n(1:size(s,1)-1), n(end)], [1,1,size(s,1)]), [length(array.el_xc), size(s,2), 1]), 3);
d = sqrt(sum(dd .^ 2, 3));
couplant_angs = acos(dd_dot_n ./ d);
couplant_time = d / immersion_options.couplant_velocity;

%loop over elements and find min time for each, checking that ends of
%surface are not used
for ii = 1:length(array.el_xc)
    total_time = sample_time + repmat(couplant_time(ii, :), [size(p, 1),1]);
    [min_time, jj] = min(total_time, [], 2);
    amp = double((jj > 1) & (jj < length(surface.x)));
    if immersion_options.angle_limit_on && immersion_options.angle_limit > 0
        amp = amp .* (abs(couplant_angs(ii, jj)) <= immersion_options.angle_limit)'; 
        %note there is no distance compensation for fixed angle here as it
        %is not clear how to do it (cf. contact TFM)
    end
    if use_coarse_mesh
        coarse_amp = reshape(amp, size(coarse_mesh.x));
        coarse_time = reshape(min_time, size(coarse_mesh.x));
        lookup_amp(:,:,ii) = interp2(coarse_mesh.x, coarse_mesh.z, coarse_amp, mesh.x, mesh.z, 'linear');
        lookup_time(:,:,ii) = interp2(coarse_mesh.x, coarse_mesh.z, coarse_time, mesh.x, mesh.z, 'linear');
    else
        lookup_amp(:,:,ii) = reshape(amp, size(mesh.x));
        lookup_time(:,:,ii) = reshape(min_time, size(mesh.x));
    end
end
end
%--------------------------------------------------------------------------

% function [lookup_time, lookup_amp] = fn_all_points_method(mesh, array, surface, sample_velocity, immersion_options)
% %pure number crunching baseline - probably the one to do on GPU -
% %find minimum travel time to each pixel in turn
% lookup_amp = zeros(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
% lookup_time = zeros(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
% for i1 = 1:size(mesh.x,1)
%     for i2 = 1:size(mesh.x,2)
%         if mesh.valid_pts(i1,i2)
%             sample_time = sqrt((mesh.x(i1,i2) - surface.x) .^ 2 + (mesh.z(i1,i2) - surface.z) .^ 2) / sample_velocity;
%             for i3 = 1:length(array.el_xc)
%                 focal_law.lookup_amp(i1,i2,i3) = 1;
%                 couplant_time = sqrt((array.el_xc(i3) - surface.x) .^ 2 + (array.el_zc(i3) - surface.z) .^ 2) / immersion_options.couplant_velocity;
%                 t = couplant_time + sample_time;
%                 [min_t, ii] = min(t);
%                 lookup_time(i1,i2,i3) = min_t;
%                 if (ii > 1) & (ii < length(surface.x))
%                     lookup_amp(i1,i2,i3) = 1;
%                 end
%             end
%         end
%     end
% end
% end

% function [lookup_time, lookup_amp] = fn_interp_method(mesh, array, surface, sample_velocity, immersion_options)
% %this function assumes all mesh pts are fully below lowest point of surface
% %perimeter points (denote x_i, z_i)
% % tic
% x1 = min(mesh.x(1,:)) - sample_wavelength;
% z1 = mesh.z(1,1);
% x2 = min(mesh.x(1,:)) - sample_wavelength;
% z2 = max(mesh.z(:,1));
% x3 = max(mesh.x(1,:)) + sample_wavelength;
% z3 = max(mesh.z(:,1));
% x4 = max(mesh.x(1,:)) + sample_wavelength;
% z4 = mesh.z(1,1);
% n12 = ceil(abs(z2-z1) / sample_wavelength * immersion_options.perim_pts_per_sample_wavelength);
% n23 = ceil(abs(x3-x2) / sample_wavelength * immersion_options.perim_pts_per_sample_wavelength);
% n34 = ceil(abs(z4-z3) / sample_wavelength * immersion_options.perim_pts_per_sample_wavelength);
% x_i = [ones(1, n12) * x1, linspace(x2, x3, n23), ones(1, n34) * x3];
% z_i = [linspace(z1, z2, n12), ones(1, n23) * z2, linspace(z3, z4, n34)];
%
% m_ij_x = zeros(length(array.el_xc), length(x_i));
% m_ij_t = zeros(length(array.el_xc), length(x_i));
% c_ij_x = zeros(length(array.el_xc), length(x_i));
% c_ij_t = zeros(length(array.el_xc), length(x_i));
% for ii = 1:length(x_i)
%     t_sample = sqrt((x_i(ii) - surface.x) .^ 2 + (z_i(ii) - surface.z) .^ 2) / sample_velocity;
%     for i3 = 1:length(array.el_xc)
%         t_couplant = sqrt((array.el_xc(i3) - surface.x) .^ 2 + (array.el_zc(i3) - surface.z) .^ 2) / immersion_options.couplant_velocity;
%         tt = t_sample + t_couplant;
%         [t_i, si] = min(tt);
%         t_s = t_couplant(si);
%         x_s = surface.x(si);
%         z_s = surface.z(si);
%         m_ij_x(i3, ii) = (x_i(ii) - x_s) / (z_i(ii) - z_s);
%         c_ij_x(i3, ii) = x_i(ii) - m_ij_x(i3, ii) * z_i(ii);
%         if (si > 1) & (si < length(surface.x))
%             m_ij_t(i3, ii) = (t_i - t_s) / (z_i(ii) - z_s);
%             c_ij_t(i3, ii) = t_i - m_ij_t(i3, ii) * z_i(ii);
%         else
%             %flag dodgy rays that go through ends of defined surface
%             m_ij_t(i3, ii) = NaN;
%             c_ij_t(i3, ii) = NaN;
%         end
%     end
% end
% % fprintf('    %.3f secs to Fermat calculation at %i pts\n', toc, length(x_i));
%
% %now work through mesh by z-depth, interpolating as nesc
% % tic;
% tmp_z = linspace(min(z_i), max(mesh.z(:,1)), ceil((max(mesh.z(:,1)) - min(z_i)) / sample_wavelength * immersion_options.pre_interp_pts_per_sample_wavelength));
% lookup_time = zeros(length(tmp_z), size(mesh.x, 2), length(array.el_xc));
% for i1 = 1:length(tmp_z)
%     tmp_x = m_ij_x * tmp_z(i1) + c_ij_x;
%     tmp_t = m_ij_t * tmp_z(i1) + c_ij_t;
%     for i3 = 1:length(array.el_xc)
%         [dummy, ii] = unique(tmp_x(i3, :));
%         warning('off');
%         lookup_time(i1,:,i3) = interp1(tmp_x(i3, ii), tmp_t(i3, ii), mesh.x(1,:), 'linear', 0);
%     end
% end
% lookup_time = interp1(tmp_z, lookup_time, mesh.z(:,1), 'linear', 0);
% warning('on');
% lookup_amp = ones(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
% ii = find(isnan(lookup_time));
% lookup_time(ii) = 0;
% lookup_amp(ii) = 0;
% % fprintf('    %.3f secs for interpolation stage\n', toc)
% end
%
% function [lookup_time, lookup_amp] = fn_interp2_method(mesh, array, surface, couplant_velocity, sample_velocity, immersion_options, use_gpu_if_available)
% %this function assumes all mesh pts are fully below lowest point of surface
% %perimeter points (denote x_i, z_i)
% tic
% x1 = min(mesh.x(1,:)) - sample_wavelength;
% z1 = mesh.z(1,1);
% x2 = min(mesh.x(1,:)) - sample_wavelength;
% z2 = max(mesh.z(:,1));
% x3 = max(mesh.x(1,:)) + sample_wavelength;
% z3 = max(mesh.z(:,1));
% x4 = max(mesh.x(1,:)) + sample_wavelength;
% z4 = mesh.z(1,1);
% n12 = ceil(abs(z2-z1) / sample_wavelength * immersion_options.perim_pts_per_sample_wavelength);
% n23 = ceil(abs(x3-x2) / sample_wavelength * immersion_options.perim_pts_per_sample_wavelength);
% n34 = ceil(abs(z4-z3) / sample_wavelength * immersion_options.perim_pts_per_sample_wavelength);
% x_i = [ones(1, n12) * x1, linspace(x2, x3, n23), ones(1, n34) * x3];
% z_i = [linspace(z1, z2, n12), ones(1, n23) * z2, linspace(z3, z4, n34)];
%
% no_dims = 2;
% a0 = zeros(no_dims, length(x_i));
% a1 = zeros(no_dims, length(x_i));
% t0 = zeros(1, length(x_i));
% t1 = zeros(1, length(x_i));
%
% % tmp_z = linspace(min(z_i), max(mesh.z(:,1)), ceil((max(mesh.z(:,1)) - min(z_i)) / options.lambda * options.pre_interp_pts_per_sample_wavelength));
% % lookup_time = zeros(length(tmp_z), size(mesh.x, 2), length(array.el_xc));
% lookup_time = zeros(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
% % if use_gpu_if_available && (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0)
% %     p = gpuArray([mesh.x(:), mesh.z(:)]');
% % else
% p = [mesh.x(:), mesh.z(:)]';
% % end
%
% for i3 = 1:length(array.el_xc) %outer loop is over array elements
%     t_couplant = sqrt((array.el_xc(i3) - surface.x) .^ 2 + (array.el_zc(i3) - surface.z) .^ 2) / immersion_options.couplant_velocity;
%     for ii = 1:length(x_i) %this loop over perimeter points
%         t_sample = sqrt((x_i(ii) - surface.x) .^ 2 + (z_i(ii) - surface.z) .^ 2) / sample_velocity;
%         tt = t_sample + t_couplant;
%         [t_total, si] = min(tt); %si is index of surface points
%         a0(1, ii) = surface.x(si);
%         a0(2, ii) = surface.z(si);
%         a1(1, ii) = x_i(ii) - surface.x(si);
%         a1(2, ii) = z_i(ii) - surface.z(si);
%         t0(ii) = t_couplant(si);
%         t1(ii) = t_total - t0(ii);
%     end
%     tmp = sqrt(sum(a1 .^ 2));
%     t1 = t1 ./ tmp;
%     a1 = a1 ./ (ones(size(a1,1),1) * tmp);
%     %Fermat rays now calculated - now find nearest line to each pixel
%     %     if use_gpu_if_available && (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0)
%     %         a0 = gpuArray(a0);
%     %         a1 = gpuArray(a1);
%     %     end
%     [jj, q] = fn_find_nearest_lines_to_points(a0, a1, p, use_gpu_if_available);
%     lookup_time(:, :, i3) = reshape(t0(jj) + t1(jj) .* q, size(mesh.x));
% end
%
% lookup_amp = ones(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
% ii = find(isnan(lookup_time));
% lookup_time(ii) = 0;
% lookup_amp(ii) = 0;
% end
%
% function [jj, q] = fn_find_nearest_lines_to_points(a0, a1, p, use_gpu_if_available)
% % if use_gpu_if_available && (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0)
% %     keyboard
% % else
% persistent p_minus_a0 a1_big q_big d
% %this function should work for all dimensions (i.e. D = 2 or 3) of problem.
% %a0, a1 and p should all have D rows
% p_minus_a0 = permute(repmat(p,[1,1,size(a0,ndims(a0))]), [1,3,2]) - repmat(a0,[1,1,size(p,ndims(p))]);
% a1_big = repmat(a1,[1,1,size(p,ndims(p))]);
% q_big = repmat(sum(p_minus_a0 .* a1_big, 1), [size(a0,1),1,1]);
% d = squeeze(sum((p_minus_a0 - q_big .* a1_big) .^ 2, 1));
% [~, jj] = min(d);
% q = q_big(sub2ind(size(q_big), ones(size(jj)), jj, [1:length(jj)]));
% % end
% end