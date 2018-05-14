function [focal_law, surface, couplant_focal_law] = fn_calc_immersion_tfm_focal_law(exp_data, mesh, orig_surface, immersion_options, varargin)
if nargin < 5
    tfm_options = [];
else
    tfm_options = varargin{1};
end
if nargin < 6
    use_gpu_if_available = 1;
else
    use_gpu_if_available = varargin{2};
end
% default_options.sample_angle_dep_vel = 0;
% default_options.couplant_angle_limit = 0;
% default_options.sample_angle_limit = 0;
default_immersion_options.couplant_velocity = 1480;
default_immersion_options.surface_pts_per_sample_wavelength = 5;
default_immersion_options.perim_pts_per_sample_wavelength = 1;
default_immersion_options.pre_interp_pts_per_sample_wavelength = 1;
default_immersion_options.centre_freq = exp_data.array.centre_freq;
default_immersion_options.method = 'interp';
default_immersion_options.extrapolate_surface = 0;
default_immersion_options.show_couplant_only = 0;

immersion_options = fn_set_default_fields(immersion_options, default_immersion_options);
surface = orig_surface;
if isfield(tfm_options, 'centre_freq')
    immersion_options.lambda = exp_data.ph_velocity / tfm_options.centre_freq;
else
    immersion_options.lambda = exp_data.ph_velocity / exp_data.array.centre_freq;
end

%upsample surface and extrapolate if nesc
if immersion_options.extrapolate_surface
    min_x = min([mesh.x(1,:)'; surface.x(:)]);
    max_x = max([mesh.x(1,:)'; surface.x(:)]);
else
    min_x = min(surface.x(:));
    max_x = max(surface.x(:));
end
tmp_x = linspace(min_x, max_x, ceil((max_x - min_x) / immersion_options.lambda * immersion_options.surface_pts_per_sample_wavelength));
surface.z = interp1(surface.x, surface.z, tmp_x, 'spline', 'extrap');
surface.x = tmp_x;

%interpolate specified surface onto grid x values - needs to be generalised
%for 2D case to identify valid points
tmp_surf_z = interp1(surface.x(:), surface.z(:), mesh.x(1,:), 'linear', 'extrap');
mesh.valid_pts = mesh.z > ones(size(mesh.z, 1),1) * tmp_surf_z;

%partition mesh - mesh1 is completely above surface and not processed at
%all; mesh2 is the surface region; mesh3 is completely below surface
zi_12 = max(find(mesh.z(:,1) < min(surface.z)));
zi_23 = min(find(mesh.z(:,1) > (max(surface.z) + 2 * immersion_options.lambda)));

mesh1.x = mesh.x(1:zi_12,:);
mesh1.z = mesh.z(1:zi_12,:);
mesh2.x = mesh.x(zi_12 + 1: zi_23 - 1,:);
mesh2.z = mesh.z(zi_12 + 1: zi_23 - 1,:);

if ~immersion_options.show_couplant_only
    switch immersion_options.method
        case 'all points'
            mesh23.x = mesh.x(zi_12+1:end,:);
            mesh23.z = mesh.z(zi_12+1:end,:);
            mesh23.valid_pts = mesh.valid_pts(zi_12+1:end,:);
            [lookup_time23, lookup_amp23] = fn_all_points_method(mesh23, exp_data.array, surface, exp_data.ph_velocity, immersion_options);
            focal_law.lookup_time = [zeros(zi_12, size(lookup_time23, 2), size(lookup_time23, 3)); lookup_time23];
            focal_law.lookup_amp = [zeros(zi_12, size(lookup_time23, 2), size(lookup_time23, 3)); lookup_amp23];
            
        case 'interp'
            mesh3.x = mesh.x(zi_23:end,:);
            mesh3.z = mesh.z(zi_23:end,:);
            [lookup_time3, lookup_amp3] = fn_interp_method(mesh3, exp_data.array, surface, exp_data.ph_velocity, immersion_options);
            mesh2.x = mesh.x(zi_12 + 1: zi_23 - 1,:);
            mesh2.z = mesh.z(zi_12 + 1: zi_23 - 1,:);
            mesh2.valid_pts = mesh.valid_pts(zi_12 + 1: zi_23 - 1,:);
            [lookup_time2, lookup_amp2] = fn_all_points_method(mesh2, exp_data.array, surface, exp_data.ph_velocity, immersion_options);
            focal_law.lookup_time = [zeros(zi_12, size(lookup_time3, 2), size(lookup_time3, 3)); lookup_time2; lookup_time3];
            focal_law.lookup_amp = [zeros(zi_12, size(lookup_time3, 2), size(lookup_time3, 3)); lookup_amp2; lookup_amp3];
            
        case 'interp2'
            mesh3.x = mesh.x(zi_23:end,:);
            mesh3.z = mesh.z(zi_23:end,:);
            [lookup_time3, lookup_amp3] = fn_interp2_method(mesh3, exp_data.array, surface, exp_data.ph_velocity, immersion_options, use_gpu_if_available);
            mesh2.x = mesh.x(zi_12 + 1: zi_23 - 1,:);
            mesh2.z = mesh.z(zi_12 + 1: zi_23 - 1,:);
            mesh2.valid_pts = mesh.valid_pts(zi_12 + 1: zi_23 - 1,:);
            [lookup_time2, lookup_amp2] = fn_all_points_method(mesh2, exp_data.array, surface, exp_data.ph_velocity, immersion_options);
            focal_law.lookup_time = [zeros(zi_12, size(lookup_time3, 2), size(lookup_time3, 3)); lookup_time2; lookup_time3];
            focal_law.lookup_amp = [zeros(zi_12, size(lookup_time3, 2), size(lookup_time3, 3)); lookup_amp2; lookup_amp3];
    end
else
    focal_law.lookup_time = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    focal_law.lookup_amp = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
end

%now the easy bit - focal law in water which is just usual TFM
if isfield(immersion_options, 'couplant_focal_law')
    sz1 = size(immersion_options.couplant_focal_law.lookup_time);
    sz2 = size(focal_law.lookup_time);
    if (sz1(1) == sz2(1)) && (sz1(2) <= sz2(2)) && (sz1(3) == sz2(3))
        focal_law.lookup_time = immersion_options.couplant_focal_law.lookup_time;
        focal_law.lookup_amp = immersion_options.couplant_focal_law.lookup_amp;
    end
    %option to copy in previous focal law if it exists already to save time
    %check sizes match; copy over relevant portion
    couplant_focal_law = immersion_options.couplant_focal_law;
else
    if immersion_options.show_couplant_only
        mesh.valid_pts = zeros(size(mesh.x));
        zi_23 = size(mesh.x, 1);
    end
    mesh12.x = mesh.x(1:zi_23,:);
    mesh12.z = mesh.z(1:zi_23,:);
    mesh12.valid_pts = mesh.valid_pts(1:zi_23,:);
    exp_data.ph_velocity = immersion_options.couplant_velocity;
    couplant_focal_law = fn_calc_tfm_focal_law2(exp_data, mesh12, immersion_options);
    for ii = 1:size(focal_law.lookup_time, 3)
        focal_law.lookup_time(1:zi_23, :, ii) = focal_law.lookup_time(1:zi_23, :, ii) + couplant_focal_law.lookup_time(:, :, ii) .* (1 - mesh12.valid_pts);
        focal_law.lookup_amp(1:zi_23, :, ii) = focal_law.lookup_amp(1:zi_23, :, ii) + couplant_focal_law.lookup_amp(:, :, ii) .* (1 - mesh12.valid_pts);
    end
    
end



%zero the amplitudes for all points above surface
% focal_law.lookup_amp = focal_law.lookup_amp .* repmat(mesh.z < ones(size(mesh.z, 1),1) * surface.z, [1,1,size(focal_law.lookup_amp, 3)]);
dt = exp_data.time(2) - exp_data.time(1);
t0 = exp_data.time(1);
focal_law.lookup_ind = round((focal_law.lookup_time - t0 / 2) / dt);
focal_law.interpolation_method = 'nearest';

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

% focal_law.tt_ind = fn_optimise_focal_law2(focal_law, exp_data.tx, exp_data.rx);
% fprintf('    %.3f for finalisation\n', toc)
end

function [lookup_time, lookup_amp] = fn_all_points_method(mesh, array, surface, sample_velocity, immersion_options)
%pure number crunching baseline - probably the one to do on GPU -
%find minimum travel time to each pixel in turn
lookup_amp = zeros(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
lookup_time = zeros(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
for i1 = 1:size(mesh.x,1)
    for i2 = 1:size(mesh.x,2)
        if mesh.valid_pts(i1,i2)
            sample_time = sqrt((mesh.x(i1,i2) - surface.x) .^ 2 + (mesh.z(i1,i2) - surface.z) .^ 2) / sample_velocity;
            for i3 = 1:length(array.el_xc)
                focal_law.lookup_amp(i1,i2,i3) = 1;
                couplant_time = sqrt((array.el_xc(i3) - surface.x) .^ 2 + (array.el_zc(i3) - surface.z) .^ 2) / immersion_options.couplant_velocity;
                t = couplant_time + sample_time;
                [min_t, ii] = min(t);
                lookup_time(i1,i2,i3) = min_t;
                if (ii > 1) & (ii < length(surface.x))
                    lookup_amp(i1,i2,i3) = 1;
                end
            end
        end
    end
end
end

function [lookup_time, lookup_amp] = fn_interp_method(mesh, array, surface, sample_velocity, immersion_options)
%this function assumes all mesh pts are fully below lowest point of surface
%perimeter points (denote x_i, z_i)
% tic
x1 = min(mesh.x(1,:)) - immersion_options.lambda;
z1 = mesh.z(1,1);
x2 = min(mesh.x(1,:)) - immersion_options.lambda;
z2 = max(mesh.z(:,1));
x3 = max(mesh.x(1,:)) + immersion_options.lambda;
z3 = max(mesh.z(:,1));
x4 = max(mesh.x(1,:)) + immersion_options.lambda;
z4 = mesh.z(1,1);
n12 = ceil(abs(z2-z1) / immersion_options.lambda * immersion_options.perim_pts_per_sample_wavelength);
n23 = ceil(abs(x3-x2) / immersion_options.lambda * immersion_options.perim_pts_per_sample_wavelength);
n34 = ceil(abs(z4-z3) / immersion_options.lambda * immersion_options.perim_pts_per_sample_wavelength);
x_i = [ones(1, n12) * x1, linspace(x2, x3, n23), ones(1, n34) * x3];
z_i = [linspace(z1, z2, n12), ones(1, n23) * z2, linspace(z3, z4, n34)];

m_ij_x = zeros(length(array.el_xc), length(x_i));
m_ij_t = zeros(length(array.el_xc), length(x_i));
c_ij_x = zeros(length(array.el_xc), length(x_i));
c_ij_t = zeros(length(array.el_xc), length(x_i));
for ii = 1:length(x_i)
    t_sample = sqrt((x_i(ii) - surface.x) .^ 2 + (z_i(ii) - surface.z) .^ 2) / sample_velocity;
    for i3 = 1:length(array.el_xc)
        t_couplant = sqrt((array.el_xc(i3) - surface.x) .^ 2 + (array.el_zc(i3) - surface.z) .^ 2) / immersion_options.couplant_velocity;
        tt = t_sample + t_couplant;
        [t_i, si] = min(tt);
        t_s = t_couplant(si);
        x_s = surface.x(si);
        z_s = surface.z(si);
        m_ij_x(i3, ii) = (x_i(ii) - x_s) / (z_i(ii) - z_s);
        c_ij_x(i3, ii) = x_i(ii) - m_ij_x(i3, ii) * z_i(ii);
        if (si > 1) & (si < length(surface.x))
            m_ij_t(i3, ii) = (t_i - t_s) / (z_i(ii) - z_s);
            c_ij_t(i3, ii) = t_i - m_ij_t(i3, ii) * z_i(ii);
        else
            %flag dodgy rays that go through ends of defined surface
            m_ij_t(i3, ii) = NaN;
            c_ij_t(i3, ii) = NaN;
        end
    end
end
% fprintf('    %.3f secs to Fermat calculation at %i pts\n', toc, length(x_i));

%now work through mesh by z-depth, interpolating as nesc
% tic;
tmp_z = linspace(min(z_i), max(mesh.z(:,1)), ceil((max(mesh.z(:,1)) - min(z_i)) / immersion_options.lambda * immersion_options.pre_interp_pts_per_sample_wavelength));
lookup_time = zeros(length(tmp_z), size(mesh.x, 2), length(array.el_xc));
for i1 = 1:length(tmp_z)
    tmp_x = m_ij_x * tmp_z(i1) + c_ij_x;
    tmp_t = m_ij_t * tmp_z(i1) + c_ij_t;
    for i3 = 1:length(array.el_xc)
        [dummy, ii] = unique(tmp_x(i3, :));
        warning('off');
        lookup_time(i1,:,i3) = interp1(tmp_x(i3, ii), tmp_t(i3, ii), mesh.x(1,:), 'linear', 0);
    end
end
lookup_time = interp1(tmp_z, lookup_time, mesh.z(:,1), 'linear', 0);
warning('on');
lookup_amp = ones(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
ii = find(isnan(lookup_time));
lookup_time(ii) = 0;
lookup_amp(ii) = 0;
% fprintf('    %.3f secs for interpolation stage\n', toc)
end

function [lookup_time, lookup_amp] = fn_interp2_method(mesh, array, surface, couplant_velocity, sample_velocity, immersion_options, use_gpu_if_available)
%this function assumes all mesh pts are fully below lowest point of surface
%perimeter points (denote x_i, z_i)
tic
x1 = min(mesh.x(1,:)) - immersion_options.lambda;
z1 = mesh.z(1,1);
x2 = min(mesh.x(1,:)) - immersion_options.lambda;
z2 = max(mesh.z(:,1));
x3 = max(mesh.x(1,:)) + immersion_options.lambda;
z3 = max(mesh.z(:,1));
x4 = max(mesh.x(1,:)) + immersion_options.lambda;
z4 = mesh.z(1,1);
n12 = ceil(abs(z2-z1) / immersion_options.lambda * immersion_options.perim_pts_per_sample_wavelength);
n23 = ceil(abs(x3-x2) / immersion_options.lambda * immersion_options.perim_pts_per_sample_wavelength);
n34 = ceil(abs(z4-z3) / immersion_options.lambda * immersion_options.perim_pts_per_sample_wavelength);
x_i = [ones(1, n12) * x1, linspace(x2, x3, n23), ones(1, n34) * x3];
z_i = [linspace(z1, z2, n12), ones(1, n23) * z2, linspace(z3, z4, n34)];

no_dims = 2;
a0 = zeros(no_dims, length(x_i));
a1 = zeros(no_dims, length(x_i));
t0 = zeros(1, length(x_i));
t1 = zeros(1, length(x_i));

% tmp_z = linspace(min(z_i), max(mesh.z(:,1)), ceil((max(mesh.z(:,1)) - min(z_i)) / options.lambda * options.pre_interp_pts_per_sample_wavelength));
% lookup_time = zeros(length(tmp_z), size(mesh.x, 2), length(array.el_xc));
lookup_time = zeros(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
% if use_gpu_if_available && (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0)
%     p = gpuArray([mesh.x(:), mesh.z(:)]');
% else
p = [mesh.x(:), mesh.z(:)]';
% end

for i3 = 1:length(array.el_xc) %outer loop is over array elements
    t_couplant = sqrt((array.el_xc(i3) - surface.x) .^ 2 + (array.el_zc(i3) - surface.z) .^ 2) / immersion_options.couplant_velocity;
    for ii = 1:length(x_i) %this loop over perimeter points
        t_sample = sqrt((x_i(ii) - surface.x) .^ 2 + (z_i(ii) - surface.z) .^ 2) / sample_velocity;
        tt = t_sample + t_couplant;
        [t_total, si] = min(tt); %si is index of surface points
        a0(1, ii) = surface.x(si);
        a0(2, ii) = surface.z(si);
        a1(1, ii) = x_i(ii) - surface.x(si);
        a1(2, ii) = z_i(ii) - surface.z(si);
        t0(ii) = t_couplant(si);
        t1(ii) = t_total - t0(ii);
    end
    tmp = sqrt(sum(a1 .^ 2));
    t1 = t1 ./ tmp;
    a1 = a1 ./ (ones(size(a1,1),1) * tmp);
    %Fermat rays now calculated - now find nearest line to each pixel
    %     if use_gpu_if_available && (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0)
    %         a0 = gpuArray(a0);
    %         a1 = gpuArray(a1);
    %     end
    [jj, q] = fn_find_nearest_lines_to_points(a0, a1, p, use_gpu_if_available);
    lookup_time(:, :, i3) = reshape(t0(jj) + t1(jj) .* q, size(mesh.x));
end

lookup_amp = ones(size(mesh.x, 1), size(mesh.x, 2), length(array.el_xc));
ii = find(isnan(lookup_time));
lookup_time(ii) = 0;
lookup_amp(ii) = 0;
end

function [jj, q] = fn_find_nearest_lines_to_points(a0, a1, p, use_gpu_if_available)
% if use_gpu_if_available && (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0)
%     keyboard
% else
persistent p_minus_a0 a1_big q_big d
%this function should work for all dimensions (i.e. D = 2 or 3) of problem.
%a0, a1 and p should all have D rows
p_minus_a0 = permute(repmat(p,[1,1,size(a0,ndims(a0))]), [1,3,2]) - repmat(a0,[1,1,size(p,ndims(p))]);
a1_big = repmat(a1,[1,1,size(p,ndims(p))]);
q_big = repmat(sum(p_minus_a0 .* a1_big, 1), [size(a0,1),1,1]);
d = squeeze(sum((p_minus_a0 - q_big .* a1_big) .^ 2, 1));
[~, jj] = min(d);
q = q_big(sub2ind(size(q_big), ones(size(jj)), jj, [1:length(jj)]));
% end
end