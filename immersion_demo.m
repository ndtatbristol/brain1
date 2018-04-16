%fn_clear;
%clear;

%test on aluminium sample with 'weld' cap profile and SHDs
exp_fname = exp_data;

%true surface file
surf_fname = 'N:\ndt-library\arrays\True surface profile for immersion example.txt';

%surface details for reconstruction
surf_type = 'Measure';
% surf_type = 'Flat';
% surf_type = 'Curved';
% surf_type = 'True';

%final image - this is the one that is ultimately displayed (covering both
%couplant and sample)
final_image_size_x = 50e-3;
final_image_size_z = 90e-3;
final_image_offset_z = 10e-3;
final_pixel_size = 0.5e-3;

db_scale = 40;
sample_velocity = 6420;
couplant_velocity = 1480;
use_gpu_if_available = 1;

%surface-finding stage parameters
surface_finding_options.centre_freq = 3e6;
surface_finding_options.frac_bandwidth = 1.5;
surface_finding_options.number_of_wavelengths_to_smooth_over = 3;
surface_finding_options.max_jump_in_wavelengths = 3;
surface_finding_options.lo_res_pts_per_lambda = 0.5;
surface_finding_options.hi_res_pts_per_lambda = 10;
surface_finding_options.couplant_velocity = couplant_velocity;
%surface-finding stage - things that should be calculated automatically
surface_finding_options.angle_limit = 30 * pi / 180;

%main imaging stage parameters - first ones are usual TFM ones
immersion_options.centre_freq = 3e6;
immersion_options.frac_bandwidth = 1.5;
immersion_options.filter_on = 1;
immersion_options.angle_limit_on = 0;
%extra options specific to immersion case
immersion_options.extrapolate_surface = 0;
immersion_options.interp_pts_per_sample_wavelength = 0.5;
immersion_options.surface_pts_per_sample_wavelength = 5;
immersion_options.couplant_velocity = couplant_velocity;
immersion_options.interpolation_method = 'linear';

%--------------------------------------------------------------------------
%load files
load(exp_fname);
true_surf_data = load(surf_fname);

%generate true surface data
true_surface.x = true_surf_data(1,:)';
true_surface.z = true_surf_data(2,:)';

%set up final image coordinates
mesh.x = linspace(-final_image_size_x /2, final_image_size_x / 2, ceil(final_image_size_x / final_pixel_size));
mesh.z = linspace(0, final_image_size_z, ceil(final_image_size_z / final_pixel_size)) + final_image_offset_z;
[mesh.x, mesh.z] = meshgrid(mesh.x, mesh.z);

%--------------------------------------------------------------------------
%Get surface
separate_calc_for_couplant_image = 1;
switch surf_type
    case 'Measure'
        surface_finding_options.use_gpu_if_available = use_gpu_if_available;
        tic;
        [surface.x, surface.z, tfm_couplant_result] = fn_extract_surface_from_immersion_data2(exp_data, mesh, surface_finding_options);
        fprintf('Time to find surface: %.3f secs\n', toc);
        separate_calc_for_couplant_image = 0;
    case 'Flat'
        flat_surf_depth = true_surface.z(136);
        surface.x = [min(true_surface.x), max(true_surface.x)];
        surface.z = flat_surf_depth * [1, 1];
    case 'Curved'
        curved_surf_rad = 0.04;
        curved_surf_xc = true_surface.x(136);
        curved_surf_zc = curved_surf_rad + true_surface.z(136);
        a = asin(abs(max(true_surface.x) - curved_surf_xc) / curved_surf_rad);
        a = linspace(-a, a, ceil(2 * a * curved_surf_rad / (exp_data.ph_velocity / exp_data.array.centre_freq)));
        surface.x = curved_surf_xc + curved_surf_rad * sin(a);
        surface.z = curved_surf_zc - curved_surf_rad * cos(a);
    case 'True'
        surface = true_surface;
end

%--------------------------------------------------------------------------
%Generate couplant focal law and image
if separate_calc_for_couplant_image
    tmp = exp_data.ph_velocity;
    exp_data.ph_velocity = immersion_options.couplant_velocity;
    couplant_focal_law = fn_calc_tfm_focal_law2(exp_data, mesh, immersion_options);
    couplant_focal_law.filter_on = 1;
    couplant_focal_law.filter = fn_calc_filter(exp_data.time, immersion_options.centre_freq, immersion_options.centre_freq * immersion_options.frac_bandwidth / 2);
    tfm_couplant_result = fn_fast_DAS2(exp_data, couplant_focal_law, use_gpu_if_available);
    exp_data.ph_velocity = tmp;
end

%--------------------------------------------------------------------------
%Generate immersion focal law
tic;
[tfm_immersion_focal_law, surface] = fn_calc_immersion_tfm_focal_law2(exp_data, mesh, surface, immersion_options, use_gpu_if_available);
%calc filter for TFM
tfm_immersion_focal_law.filter = fn_calc_filter(exp_data.time, immersion_options.centre_freq, immersion_options.centre_freq * immersion_options.frac_bandwidth / 2);
tfm_immersion_focal_law.filter_on = immersion_options.filter_on;
tfm_immersion_focal_law.interpolation_method = 'linear';
% immersion_options.couplant_focal_law = couplant_focal_law; %use this if repear calls to above function are made to save time
fprintf('Time to generate immersion focal law: %.3f secs\n', toc);

%--------------------------------------------------------------------------
%Generate sample image
tic;
tfm_immersion_result = fn_fast_DAS2(exp_data, tfm_immersion_focal_law, use_gpu_if_available);
fprintf('Time to generate final image: %.3f secs\n', toc);

%--------------------------------------------------------------------------
%Generate merged image
sample_pts = sum(tfm_immersion_focal_law.lookup_amp, 3) > 0;
tfm_immersion_result = tfm_immersion_result .* sample_pts;
tfm_couplant_result = tfm_couplant_result .* (1 - sample_pts);
tfm_couplant_result = tfm_couplant_result / max(max(max(abs(tfm_couplant_result)))) * max(max(max(abs(tfm_immersion_result))));
merged_result = tfm_immersion_result + tfm_couplant_result;

%--------------------------------------------------------------------------
%Display images
figure;
tmp = abs(tfm_couplant_result);
tmp = tmp / max(max(tmp));
imagesc(mesh.x(1,:), mesh.z(:,1), 20*log10(tmp));
caxis([-40, 0]);
hold on;
plot(surface.x, surface.z, 'w');
plot(surface.x, surface.z, 'r:');
plot(true_surface.x, true_surface.z, 'g');
plot(true_surface.x, true_surface.z, 'y:');
axis equal; axis tight;
title('Couplant result');

figure;
tmp = abs(tfm_immersion_result);
tmp = tmp / max(max(tmp));
imagesc(mesh.x(1,:), mesh.z(:,1), 20*log10(tmp));
caxis([-40, 0]);
hold on;
plot(surface.x, surface.z, 'w');
plot(surface.x, surface.z, 'r:');
plot(true_surface.x, true_surface.z, 'g');
plot(true_surface.x, true_surface.z, 'y:');
axis equal; axis tight;
title('Sample result');

figure;
tmp = abs(merged_result);
tmp = tmp / max(max(tmp));
imagesc(mesh.x(1,:), mesh.z(:,1), 20*log10(tmp));
caxis([-40, 0]);
hold on;
plot(surface.x, surface.z, 'w');
plot(surface.x, surface.z, 'r:');
plot(true_surface.x, true_surface.z, 'g');
plot(true_surface.x, true_surface.z, 'y:');
axis equal; axis tight;
title('Merged result');
