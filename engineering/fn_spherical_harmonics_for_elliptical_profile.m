function f_lm = fn_spherical_harmonics_for_elliptical_profile(x_velocity, y_velocity, z_velocity, lebedev_quality)
%SUMMARY
%   Returns spherical harmonic coefficients matrix for the case of a
%   anisotropic velocity profile that is ellipspoidal and aligned with x y
%   and z axes.
%INPUTS
%   x_velocity - velocity in x direction(scalar)
%   y_velocity - velocity in y direction(scalar)
%   z_velocity - velocity in z direction (scalar)
%   lebedev_quality - number between 1 (low) and 5 (max) of quality of fit.
%   Higher quality takes longer and will return larger matrix of spherical
%   harmonics.
%OUTPUTS
%   f_lm - matrix (column vector in this case) of spherical harmonics. IN
%   general: row = degree starting at 0; colsare order from -max(degree) to
%   +max(degree) with 0-order in the central column. In this case the
%   profile is axi-symmetric in the azimuthal direction so there are only
%   zero-order coefficients.
%--------------------------------------------------------------------------

%special case for isotropic
if (x_velocity == z_velocity) && (x_velocity == y_velocity)
    f_lm = x_velocity;
    return
end

[phi, theta, weight, max_order] = fn_lebedev_quadrature(lebedev_quality);

% v = xy_velocity * z_velocity ./ sqrt((xy_velocity * cos(theta)) .^ 2 + (z_velocity * sin(theta)) .^ 2);
v = x_velocity * y_velocity * z_velocity ./ ...
    sqrt((z_velocity * sin(theta)) .^ 2 .* (y_velocity ^ 2 * cos(phi) .^ 2 + x_velocity ^ 2 * sin(phi) .^ 2) + (x_velocity * y_velocity * cos(theta)) .^ 2);

degree = [0: round(max_order / 2)]; %gives pretty close to minimum error for given number of Lebedev points
order = [-degree(end): degree(end)];

if x_velocity == y_velocity %special case for transversely isotropic material in xy
    order = 0;
end

% v_check = zeros(size(v));
f_lm = zeros(length(degree), length(order));
% current_error = zeros(length(degree), 1);
for r = 1:length(degree) %row
    current_degree = degree(r); %current degree
    [sph_harm, current_orders] = fn_spherical_harmonics(phi, theta, current_degree);
    for i = 1: length(current_orders)
        c = find(order == current_orders(i)); %column associated with order
        if ~isempty(c)
            f_lm(r,c) = sum(v .* conj(squeeze(sph_harm(:, 1, i))) .* weight);%this is the integration
%             v_check = v_check + squeeze(sph_harm(:, 1, i)) .* f_lm(r, c);
        end
    end
    %calculate error at current level
%     current_error(r) = sqrt(sum(abs(v - real(v_check)) .^ 2));
end

end