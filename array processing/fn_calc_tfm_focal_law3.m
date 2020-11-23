function focal_law = fn_calc_tfm_focal_law3(exp_data, mesh, varargin)
%USAGE
%   focal_law = fn_calc_focal_law3(exp_data, mesh, [options])
%SUMMARY
%   Computes the focal law for TFM for use with fn_fast_DAS3. This is the
%   gneral 2D or 3D version depending on fields of mesh, includes smooth apertures rather than just binary (x and z; x,y and z)
%INPUTS
%   exp_data - a representative set of experimental data, from which the
%   array geometry, time axis etc will be derived
%   mesh.x, [mesh.y,] mesh.z - matrices of image point coordinates (typically created using
%   meshgrid command)
%   options [] - optional structured variable of special options, such as
%   limited aperture angle and forcing use of angle dependent velocity
%   options.transmit_mode - for FMC/HMC/SAFT, function works automatically,
%   but for other transmit modes this option will alter focal law. At
%   present, 'CSM' is only alternative.
%OUTPUTS
%   focal_law - all that's necessary to use with fn_fast_DAS function
%NOTES
%   if options.angle_limit is non-zero then aperture angle at every image
%   point (i.e. f-number) is limited

if nargin < 3
    options = [];
else
    options = varargin{1};
end
default_options.angle_dep_vel = 0;
default_options.angle_limit = 0;
default_options.interpolation_method = 'linear';
default_options.load_kernel = 1;
default_options.angle_limit_window = 'rectangular'; %options are 'hanning' or 'rectangular' for a hard edge window
default_options.weighting = 'none';%options are 'resolution', 'detection' or 'none';
default_options.aperture_weight_correction = 'weights';
default_options.transmit_mode = 'FMC/HMC';

options = fn_set_default_fields(options, default_options);

if isfield(mesh, 'y')
    n = 3;
else
    n = 2;
end
if n == 2
    dist = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    theta = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    depth = zeros(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    for ii = 1:length(exp_data.array.el_xc)
        dist(:, :, ii) = sqrt((mesh.x - exp_data.array.el_xc(ii)) .^ 2 + (mesh.z - exp_data.array.el_zc(ii)) .^ 2);
        theta(:, :, ii) = atan2(mesh.x - exp_data.array.el_xc(ii), mesh.z - exp_data.array.el_zc(ii));
        depth(:, :, ii) = abs(mesh.z - exp_data.array.el_zc(ii));
    end
    phi = 0; %azimuthal angle in xy plane is 0 for 2D imaging with 1D array
else
    dist = zeros(size(mesh.x, 1), size(mesh.x, 2),  size(mesh.x, 3), length(exp_data.array.el_xc));
    theta = zeros(size(mesh.x, 1), size(mesh.x, 2),  size(mesh.x, 3), length(exp_data.array.el_xc));
    phi = zeros(size(mesh.x, 1), size(mesh.x, 2),  size(mesh.x, 3), length(exp_data.array.el_xc));
    depth = zeros(size(mesh.x, 1), size(mesh.x, 2),  size(mesh.x, 3), length(exp_data.array.el_xc));
    for ii = 1:length(exp_data.array.el_xc)
        dist(:, :, :, ii) = sqrt((mesh.x - exp_data.array.el_xc(ii)) .^ 2 + (mesh.y - exp_data.array.el_yc(ii)) .^ 2 + (mesh.z - exp_data.array.el_zc(ii)) .^ 2);
        theta(:, :, :, ii) = atan2(sqrt((mesh.x - exp_data.array.el_xc(ii)) .^ 2 + (mesh.y - exp_data.array.el_yc(ii)) .^ 2), mesh.z - exp_data.array.el_zc(ii));
        phi(:, :, :, ii) = atan2(mesh.y - exp_data.array.el_yc(ii), mesh.x - exp_data.array.el_xc(ii));
        depth(:, :, :, ii) = abs(mesh.z - exp_data.array.el_zc(ii));
    end
end

dt = exp_data.time(2) - exp_data.time(1);
t0 = exp_data.time(1);

if options.angle_dep_vel && (... %angle dependent velocity
        isfield(exp_data, 'material') || ... %as spherical harmonics
        isfield(exp_data, 'vel_elipse') || ... %as ellipse
        isfield(exp_data, 'vel_poly')) %as polynomical
    if isfield(exp_data, 'material') %new files should contain this
        v = fn_vel_from_spherical_harmonics(exp_data.material.vel_spherical_harmonic_coeffs, phi, theta);
    elseif isfield(exp_data, 'vel_elipse') %legacy elliptical transversely-isotropic velocity profile has next priority
        v = exp_data.vel_elipse(1) * exp_data.vel_elipse(2) ./ ...
        sqrt(...
            exp_data.vel_elipse(2) .^ 2 .* cos(abs(theta)) .^ 2 + ...
            exp_data.vel_elipse(1) .^ 2 .* sin(abs(theta)) .^ 2);
    elseif isfield(exp_data, 'vel_poly') %legacy polynomial transversely-isotropic velocity profile has lowest priority
        v = polyval(exp_data.vel_poly.p, abs(theta), [], exp_data.vel_poly.mu);
    else
        warning('No valid angle-dependent velocity description found');
        if isfield(exp_data, 'ph_velocity')
            v = exp_data.ph_velocity; %legacy files
        else
            error('No valid velocity description found');
        end
    end
else %fixed velocity
    if isfield(exp_data, 'material')
        [v, ~, ~, ~] = fn_get_nominal_velocity(exp_data.material.vel_spherical_harmonic_coeffs);
    elseif isfield(exp_data, 'ph_velocity')
        v = exp_data.ph_velocity; %legacy files
    else
        error('No valid velocity description found');
    end
end

if length(options.angle_limit) == 1
    options.angle_limit = [options.angle_limit, 0, 0]; % order is [limit, look_elevation, look_azimuthal]
end

focal_law.lookup_time = dist ./ v;
focal_law.lookup_ind = round((focal_law.lookup_time - t0 / 2) / dt);

if abs(options.angle_limit(1)) %limit angle has to be > 0 for any sort of angle limit
    %work with dot products of unit vectors - easier than spherical angles
    %look angles into look vector (lv) and dot product to get gamma - angle between
    %ray and look direction
    if n == 2
        lv = [...
            sin(options.angle_limit(2)), ...
            cos(options.angle_limit(2))];
        gamma = acos(...
            sin(theta) * lv(1) + ...
            cos(theta) * lv(2));
    else
        lv = [...
            sin(options.angle_limit(3)) * sin(options.angle_limit(2)), ...
            cos(options.angle_limit(3)) * sin(options.angle_limit(2)), ...
                                          cos(options.angle_limit(2))];
        gamma = acos(...
            cos(phi) .* sin(theta) * lv(1) + ...
            sin(phi) .* sin(theta) * lv(2) + ...
                        cos(theta) * lv(3));
    end
    switch options.angle_limit_window
        case 'hanning'
            focal_law.lookup_amp = (cos(gamma .* pi ./ options.angle_limit(1)) + 1) / 2 .* (abs(gamma) <= options.angle_limit(1));
        case 'rectangular'
            focal_law.lookup_amp = (abs(gamma) <= options.angle_limit(1));
        otherwise
            error('Unknown angle limit window type');
    end
    
    %correct for varying number of contributions to different points
    switch options.aperture_weight_correction
        case 'none'
            %do nothing!
        case 'weights'
            if n == 2
                focal_law.lookup_amp = focal_law.lookup_amp ./ ...
                    repmat(sum(focal_law.lookup_amp, 3), [1, 1, size(focal_law.lookup_amp, 3)]) .* size(focal_law.lookup_amp, 3);
            else
                focal_law.lookup_amp = focal_law.lookup_amp ./ ...
                    repmat(sum(focal_law.lookup_amp, 4), [1, 1, 1, size(focal_law.lookup_amp, 4)]) .* size(focal_law.lookup_amp, 4);
            end
        case 'distance'
                focal_law.lookup_amp = focal_law.lookup_amp ./ depth;
    end
else %no angle limit
    if n == 2
        focal_law.lookup_amp = ones(size(mesh.x, 1), size(mesh.x, 2), length(exp_data.array.el_xc));
    else
        focal_law.lookup_amp = ones(size(mesh.x, 1), size(mesh.x, 2), size(mesh.x, 3), length(exp_data.array.el_xc));
    end
end

switch options.weighting
    case 'resolution'
        focal_law.lookup_amp = focal_law.lookup_amp ./ cos(theta) .* sqrt(dist);
    case 'detection'
        focal_law.lookup_amp = focal_law.lookup_amp .* cos(theta) ./ sqrt(dist);
end

focal_law.lookup_amp(isnan(focal_law.lookup_amp)) = 0;
focal_law.filter_on = 0;
focal_law.hilbert_on = 1;
focal_law.tt_weight = ones(1, length(exp_data.tx));

switch options.transmit_mode %can possibly add in PWI, VS etc options here at a later date
    case 'CSM'
        focal_law.lookup_time_rx = focal_law.lookup_time;
        focal_law.lookup_ind_rx = focal_law.lookup_ind;
        focal_law.lookup_amp_rx = focal_law.lookup_amp;
        focal_law = rmfield(focal_law, 'lookup_time');
        focal_law = rmfield(focal_law, 'lookup_ind');
        focal_law = rmfield(focal_law, 'lookup_amp');
        if options.angle_dep_vel && (... %angle dependent velocity
                isfield(exp_data, 'material') || ... %as spherical harmonics
                isfield(exp_data, 'vel_elipse') || ... %as ellipse
                isfield(exp_data, 'vel_poly')) %as polynomical
            if isfield(exp_data, 'material') %new files should contain this
                v = fn_vel_from_spherical_harmonics(exp_data.material.vel_spherical_harmonic_coeffs, 0, 0);
            elseif isfield(exp_data, 'vel_elipse') %legacy elliptical transversely-isotropic velocity profile has next priority
                v = exp_data.vel_elipse(2);
            elseif isfield(exp_data, 'vel_poly') %legacy polynomial transversely-isotropic velocity profile has lowest priority
                v = polyval(exp_data.vel_poly.p, 0, [], exp_data.vel_poly.mu);
            else
                warning('No valid angle-dependent velocity description found');
                if isfield(exp_data, 'ph_velocity')
                    v = exp_data.ph_velocity; %legacy files
                else
                    error('No valid velocity description found');
                end
            end
        else %fixed velocity
            if isfield(exp_data, 'material')
                [v, ~, ~, ~] = fn_get_nominal_velocity(exp_data.material.vel_spherical_harmonic_coeffs);
            elseif isfield(exp_data, 'ph_velocity')
                v = exp_data.ph_velocity; %legacy files
            else
                error('No valid velocity description found');
            end
        end
        focal_law.lookup_time_tx = depth / v;
        focal_law.lookup_ind_tx = round((focal_law.lookup_time_tx - t0 / 2) / dt);
        if n == 2
            focal_law.lookup_amp_tx = ...
                (mesh.x >= min(exp_data.array.el_xc)) & ...
                (mesh.x <= max(exp_data.array.el_xc));
        else
            focal_law.lookup_amp_tx = ...
                (mesh.x >= min(exp_data.array.el_xc)) & ...
                (mesh.x <= max(exp_data.array.el_xc)) & ...
                (mesh.y >= min(exp_data.array.el_yc)) & ...
                (mesh.y <= max(exp_data.array.el_yc));
        end
        focal_law.lookup_amp_tx = repmat(focal_law.lookup_amp_tx, [1,1,length(exp_data.array.el_xc)]);
    otherwise
        %add time-trace weighting vector to account for missing pitch-catch data
        %if half matrix capture is used
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

focal_law.interpolation_method = options.interpolation_method;

focal_law.tt_ind = fn_optimise_focal_law2(focal_law, exp_data.tx, exp_data.rx);

if options.load_kernel
    focal_law = fn_load_kernel(focal_law);
end

return