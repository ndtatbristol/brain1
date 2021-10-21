%Example of creating an anisotopric velocity profile directly and using
%spherical harmonic coefficients, and comparing results
close all
lebedev_quality = 3;

%Final result after conversion back from spherical harmonics will be
%calculated at uniformly spaced points in spherical coordinates
ang_pts_for_output = 72;

case_to_do = 'from stiffness matrix';
vph_or_vgr = 'vgr';
mode_no = 3;

case_to_do = 'elliptical';
x_velocity = 3000;
y_velocity = 3000;
z_velocity = 6000;

%Values for CMSX-4 from Kit Lane's thesis
c11 = 235e9;
c12 = 142e9;
c44 = 131e9;
rho = 8720;
C = [c11, c12, c12, 0, 0, 0
    c12, c11, c12, 0, 0, 0
    c12, c12, c11, 0, 0, 0
    0, 0, 0, c44, 0, 0
    0, 0, 0, 0, c44, 0
    0, 0, 0, 0, 0, c44];

%--------------------------------------------------------------------------

phi = linspace(-pi,pi,ang_pts_for_output + 1);
theta = linspace(0,pi, round(ang_pts_for_output / 2) + 1);
[PHI, THETA] = meshgrid(phi, theta);

switch case_to_do
    case 'from stiffness matrix'
        %Calculate velocity directly for reference
        [x,y,z] = sph2cart(PHI, pi / 2 - THETA, 1);
        n = [x(:), y(:), z(:)];
        
        [vph, vgr, p] = fn_anisotropic_vel_profile(C, rho, n);
        switch vph_or_vgr
            case 'vph'
                v_ref = sqrt(sum(vph(:,:,mode_no) .^ 2, 2));
            otherwise
                v_ref = sqrt(sum(vgr(:,:,mode_no) .^ 2, 2));
        end
        v_ref = reshape(v_ref, size(PHI));
        
        %Get result directly in spherical harmonics
        f_lm = fn_spherical_harmonics_from_stiffness_matrix(C, rho, mode_no, vph_or_vgr, lebedev_quality);
    case 'elliptical'
%         v_ref = xy_velocity * z_velocity ./ sqrt((xy_velocity * cos(THETA)) .^ 2 + (z_velocity * sin(THETA)) .^ 2);
        v_ref = x_velocity * y_velocity * z_velocity ./ ...
            sqrt((z_velocity * sin(THETA)) .^ 2 .* (y_velocity ^ 2 * cos(PHI) .^ 2 + x_velocity ^ 2 * sin(PHI) .^ 2) + (x_velocity * y_velocity * cos(THETA)) .^ 2);

        f_lm = fn_spherical_harmonics_for_elliptical_profile(x_velocity, y_velocity, z_velocity, lebedev_quality);
end

%Recover velocity from spherical harmonics
v = fn_vel_from_spherical_harmonics(f_lm, PHI, THETA);

figure;
subplot(2,2,1);
[Xm,Ym,Zm] = sph2cart(PHI, pi / 2 - THETA, 1);
Cm = abs(v_ref);
surf(Xm,Ym,Zm,Cm, 'EdgeColor', 'none');
axis equal; axis off;
title('Original');
colorbar

subplot(2,2,2);
[Xm,Ym,Zm] = sph2cart(PHI, pi / 2 - THETA, 1);
Cm = abs(v);
surf(Xm,Ym,Zm,Cm, 'EdgeColor', 'none');
axis equal; axis off;
title('From harmonics');
colorbar

subplot(2,2,3);
[Xm,Ym,Zm] = sph2cart(PHI, pi / 2 - THETA, 1);
Cm = v_ref - v;
surf(Xm,Ym,Zm,Cm, 'EdgeColor', 'none');
axis equal; axis off;
title('|Difference|');
colorbar

%Plot amplitude of spherical harmonics
subplot(2,2,4);
imagesc([-(size(f_lm, 2) - 1) / 2: (size(f_lm, 2) - 1) / 2], [0: size(f_lm,1) - 1], 20 * log10(abs(f_lm) / max(max(abs(f_lm(2:end, :))))));
caxis([-60,0]); colorbar;
xlabel('Order'); ylabel('Degree'); title('Spherical harmonic coefficients');
