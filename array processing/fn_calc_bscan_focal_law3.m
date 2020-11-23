function focal_law = fn_calc_bscan_focal_law3(exp_data, mesh, varargin)
%SUMMARY
%   Computes the focal law for Bscan for use with fn_fast_DAS3. This is the
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
%   focal_law - all that's necessary to use with fn_fast_DAS3 function
if nargin < 3
    options = [];
else
    options = varargin{1};
end
default_options.aperture_size_els = 8;
default_options.beam_angle = 0;
default_options.transmit_mode = 'FMC/HMC';

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
    end
else
    dist = zeros(size(mesh.x, 1), size(mesh.x, 2),  size(mesh.x, 3), length(exp_data.array.el_xc));
    for ii = 1:length(exp_data.array.el_xc)
        dist(:, :, :, ii) = mesh.z - exp_data.array.el_zc(ii);
    end
end

dt = exp_data.time(2) - exp_data.time(1);
t0 = exp_data.time(1);

if isfield(exp_data, 'material')
    [v, ~, ~, ~] = fn_get_nominal_velocity(exp_data.material.vel_spherical_harmonic_coeffs);
elseif isfield(exp_data, 'ph_velocity')
    v = exp_data.ph_velocity; %legacy files
else
    error('No valid velocity description found');
end

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
end

%normalise by number of time traces corresponding to each pixel
focal_law.lookup_amp = focal_law.lookup_amp ./ sum(focal_law.lookup_amp, n + 1);


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
        depth = mean(dist, ndims(dist)); %bit lazy - basically this only has any meaning if elements are all along z = 0 plane anyway.
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

focal_law.tt_ind = fn_optimise_focal_law2(focal_law, exp_data.tx, exp_data.rx);
focal_law.interpolation_method = 'nearest';
return