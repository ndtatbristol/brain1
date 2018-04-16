function [surf_x, surf_z] = fn_extract_surface_from_immersion_data(exp_data, options)
default_options.max_x = max(exp_data.array.el_xc);
default_options.min_x = min(exp_data.array.el_xc);
default_options.min_z = -0.5 * abs(max(exp_data.array.el_zc));
default_options.max_z =  0.5 * abs(max(exp_data.array.el_zc));
default_options.lo_res_pts_per_lambda = 0.5;
default_options.hi_res_pts_per_lambda = 20;
default_options.lo_res_x_step = abs(exp_data.array.el_xc(2) - exp_data.array.el_xc(1));
default_options.centre_freq = 3e6;
default_options.frac_bandwidth = 1.5;
default_options.angle_limit = 30 * pi / 180;
default_options.number_of_wavelengths_to_smooth_over = 3;
default_options.max_jump_in_wavelengths = 3;
default_options.use_gpu_if_available = 1;
default_options.couplant_velocity = 1480;
options = fn_set_default_fields(options, default_options);

lambda = options.couplant_velocity / options.centre_freq;
exp_data.ph_velocity = options.couplant_velocity; %needed as this is what fn_fast_DAS2 uses

%general
tfm_options.filt = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_bandwidth / 2);
tfm_options.angle_limit = options.angle_limit;
tfm_options.filter_on = 1;

%first pass - low res using all info in time-domain to find rough location
%of surface
lo_res_z_step = lambda / options.lo_res_pts_per_lambda / 2;
x = linspace(options.min_x, options.max_x, ceil((options.max_x - options.min_x) / lo_res_z_step));
z = linspace(options.min_z, options.max_z, ceil((options.max_z - options.min_z) / lo_res_z_step));
[lo_res_mesh.x, lo_res_mesh.z] = meshgrid(x, z);
focal_law_lo_res = fn_calc_tfm_focal_law2(exp_data, lo_res_mesh, tfm_options);
focal_law_lo_res.filter_on = 1;
focal_law_lo_res.filter = tfm_options.filt;
lo_res_image = fn_fast_DAS2(exp_data, focal_law_lo_res, options.use_gpu_if_available);

%find maxima in each column of image
[lo_res_surf_x, lo_res_surf_z] = fn_get_surf_from_image3(lo_res_mesh, lo_res_image, lambda, options.number_of_wavelengths_to_smooth_over, options.max_jump_in_wavelengths);

%generate hi resolution image tracking lo res maxima
hi_res_mesh.x = lo_res_surf_x;
hi_res_mesh.z = linspace(-lambda, lambda, ceil(2 * options.hi_res_pts_per_lambda))';
[hi_res_mesh.x, hi_res_mesh.z] = meshgrid(hi_res_mesh.x, hi_res_mesh.z);
hi_res_mesh.z = hi_res_mesh.z + ones(size(hi_res_mesh.z, 1),1) * lo_res_surf_z';
focal_law_hi_res = fn_calc_tfm_focal_law2(exp_data, hi_res_mesh, tfm_options);
focal_law_hi_res.filter_on = 1;
focal_law_hi_res.filter = tfm_options.filt;
hi_res_image = fn_fast_DAS2(exp_data, focal_law_hi_res, options.use_gpu_if_available);

%find maxima in each column of image
[surf_x, surf_z] = fn_get_surf_from_image3(hi_res_mesh, hi_res_image, lambda, options.number_of_wavelengths_to_smooth_over, options.max_jump_in_wavelengths);

% figure;
% tmp = abs(lo_res_image);
% tmp = tmp / max(max(tmp));
% imagesc(lo_res_mesh.x(1,:), lo_res_mesh.z(:,1), 20*log10(tmp));
% caxis([-60, 0]);
% hold on;
% plot(surf_x, surf_z, 'y.-');
% axis equal; axis tight;
end


%another method - calc moving average, remove points more than
%n*wavelengths from average and repeat until all OK
function [surf_x, surf_z] = fn_get_surf_from_image2(mesh, im, lambda, n, m)
av_pts = ceil(n * lambda / abs(mesh.x(1,2) - mesh.x(1,1)) / 2) * 2 + 1;
[max_amp, zi] = max(abs(im));
xi = 1:length(zi);
bad_pts = 1;
surf_z = mesh.z(sub2ind(size(mesh.z), zi,xi));
surf_x = mesh.x(sub2ind(size(mesh.z), zi,xi));
surf_x = surf_x(:);
surf_z = surf_z(:);
old_surf_x = surf_x;
old_surf_z = surf_z;
while ~isempty(bad_pts);
    surf_z_av = fn_moving_average(surf_z(:), av_pts);
    dz = abs(surf_z - surf_z_av);
    bad_pts = find(dz > (m * lambda));
    surf_x(bad_pts) = [];
    surf_z(bad_pts) = [];
end
surf_z = surf_z_av;
end

%another method - start from max 2 pts in adjacent cols and work outwards
function [surf_x, surf_z] = fn_get_surf_from_image3(mesh, im, lambda, n, m)
av_pts = ceil(n * lambda / abs(mesh.x(1,2) - mesh.x(1,1)) / 2) * 2 + 1;

%find largest point and its largest neighbour
[max_amps, tmp_zi] = max(abs(im));
[max_amp, xi] = max(max_amps);
if xi < size(mesh.x,2)
    xi = [xi, xi + 1];
else
    xi = [xi, xi - 1];
end
zi = tmp_zi(xi);
    
%work outwards
for dir = [-1,1]
    while (xi(end) < size(mesh.x, 2)) & (xi(end) > 1)
        if dir == -1
            current_xi = xi(1);
        else
            current_xi = xi(end);
        end
        xi_max = max([find(mesh.x(1,:) < (mesh.x(1,current_xi) + lambda * m)), current_xi + 1]);
        xi_min = min([find(mesh.x(1,:) > (mesh.x(1,current_xi) - lambda * m)), current_xi - 1]);
        ii = current_xi + dir;
        valid_pt_fnd = 0;
        while (ii <= xi_max) & (ii >= xi_min) & (ii > 0) & (ii < size(mesh.x, 2))
            xx = mesh.x(1,xi);
            zz = mesh.z(sub2ind(size(mesh.z), zi, xi));
            zz = fn_moving_average(zz(:), av_pts);
            [~, kk] = unique(xx);
            pred_z = interp1(xx(kk), zz(kk), mesh.x(1,ii) , 'spline', 'extrap');
            act_z = mesh.z(sub2ind(size(mesh.z), tmp_zi(ii), ii));
            dz = abs(pred_z - act_z);
            if dz < lambda * m
                if dir == -1
                    xi = [ii, xi];
                    zi = [tmp_zi(ii), zi];
                else
                    xi = [xi, ii];
                    zi = [zi, tmp_zi(ii)];
                end
                valid_pt_fnd = 1;
                break;
            else
                ii = ii + dir;
            end
        end
        if ~valid_pt_fnd
            break;
        end
    end
end
surf_x = mesh.x(sub2ind(size(mesh.x), zi, xi));
surf_z = mesh.z(sub2ind(size(mesh.x), zi, xi));
[surf_x, ii] = sort(surf_x);
surf_z = surf_z(ii);
surf_z = fn_moving_average(surf_z(:), av_pts);
end

