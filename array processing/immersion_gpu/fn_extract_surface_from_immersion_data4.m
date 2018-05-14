function [surf_x, surf_z] = fn_extract_surface_from_immersion_data4(exp_data, mesh, options)
default_options.min_z = 0;
default_options.max_z = max(exp_data.array.el_xc) - min(exp_data.array.el_xc);
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

tmp = sort([options.min_z; options.max_z]);
options.min_z = tmp(1);
options.max_z = tmp(2);
lambda = options.couplant_velocity / options.centre_freq;
exp_data.ph_velocity = options.couplant_velocity; %needed as this is what fn_fast_DAS2 uses

%general
tfm_options.filt = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_bandwidth / 2);
tfm_options.angle_limit = options.angle_limit;
tfm_options.filter_on = 1;

%generate coarse mesh - either the as-supplied mesh if it is of adequate
%step size of an upsampled version of it.
mesh_step = min([mesh.x(1,2) - mesh.x(1,1), mesh.z(2,1) - mesh.z(1,1)]);
max_lo_res_step = lambda / options.lo_res_pts_per_lambda / 2;

zi1 = max(find(mesh.z(:,1) < options.min_z));if isempty(zi1); zi1 = 1; end
zi2 = min(find(mesh.z(:,1) > options.max_z));if isempty(zi2); zi2 = size(mesh.x,1); end
if mesh_step <= max_lo_res_step
    lo_res_mesh = mesh;
    lo_res_mesh.x = lo_res_mesh.x(zi1:zi2, :);
    lo_res_mesh.z = lo_res_mesh.z(zi1:zi2, :);
    mesh_step_ratio = 1;
else
    mesh_step_ratio = ceil(mesh_step / max_lo_res_step);
    lo_res_step = mesh_step / mesh_step_ratio;
    x = [min(min(min(mesh.x))): lo_res_step: max(max(max(mesh.x)))];
    z = [mesh.z(zi1,1): lo_res_step: mesh.z(zi2, 1)];
    [lo_res_mesh.x, lo_res_mesh.z] = meshgrid(x, z);
end
%[result] = fn_gpu_2_stage(exp_data, lo_res_mesh, options);
lo_res_mesh.y=0;
temp_mesh=lo_res_mesh;
temp_mesh.z=temp_mesh.z;%+options.array_standoff;
exp_data.dx= lo_res_mesh.x(1,2)-lo_res_mesh.x(1,1);
exp_data.dz= lo_res_mesh.z(2)-lo_res_mesh.z(1);
exp_data.angle_limit=options.angle_limit;

entry='tfm_2d';
bit_ver=mexext;
ptx_file=['nomesh_tfm_imm' bit_ver([end-1:end]) '.ptx'];
exp_data.k=parallel.gpu.CUDAKernel(ptx_file, 'nomesh_tfm_imm.cu',entry);

lo_res_image=fn_tfm_gpu_nomesh_imm(exp_data,temp_mesh,tfm_options.filt);

%find maxima in each column of image
[lo_res_surf_x, lo_res_surf_z] = fn_get_surf_from_image2(lo_res_mesh, lo_res_image, lambda, options.number_of_wavelengths_to_smooth_over, options.max_jump_in_wavelengths);

%generate hi resolution image tracking lo res maxima
hi_res_mesh.x = lo_res_surf_x;
hi_res_mesh.z = linspace(-lambda, lambda, ceil(2 * options.hi_res_pts_per_lambda))';
[hi_res_mesh.x, hi_res_mesh.z] = meshgrid(hi_res_mesh.x, hi_res_mesh.z);
hi_res_mesh.z = hi_res_mesh.z + ones(size(hi_res_mesh.z, 1),1) * lo_res_surf_z';

temp_mesh=hi_res_mesh;
temp_mesh.y=0;
temp_mesh.z=temp_mesh.z;%+options.array_standoff;
exp_data.dx= temp_mesh.x(1,2)-temp_mesh.x(1,1);
exp_data.dz= temp_mesh.z(2)-temp_mesh.z(1);
hi_res_image=fn_tfm_gpu_nomesh_imm(exp_data,temp_mesh,tfm_options.filt);
%[hi_result] = fn_gpu_2_stage(exp_data, hi_res_mesh, options);

%wait(gpuDevice(1));
%hi_res_image=double(gather(abs(result)));

%find maxima in each column of image

[surf_x, surf_z] = fn_get_surf_from_image2(hi_res_mesh, hi_res_image, lambda, options.number_of_wavelengths_to_smooth_over, options.max_jump_in_wavelengths);

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
            pred_z = interp1(xx(kk), zz(kk), mesh.x(1,ii) , 'linear', 'extrap');%
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

