function f_lm = fn_spherical_harmonics_from_stiffness_matrix(stiffness_matrix_or_tensor, rho, mode_no, vph_or_vgr, lebedev_quality)
%SUMMARY
%   Calculates phase or group velocity profile in spherical harmonic
%   coefficients from stiffness matrix and density.
%   Recover velocity from these using fn_vel_from_spherical_harmonics.
%   Alternatively, use fn_anisotropic_vel_profile to get velocity directly
%   in specified directions.
%INPUTS
%   stiffness_matrix_or_tensor - 6x6 stiffness matrix or 3x3x3x3 tensor
%   rho - density (scalar)
%   mode_no - number between 1 and 3 of mode to return (1 and 2 are quasi-
%   shear; 3 is quasi-longitudinal)
%   vph_or_vgr - string to determin if group ('vgr') or phase ('vph') is
%   returned
%   lebedev_quality - number between 1 (low) and 5 (max) of quality of fit.
%   Higher quality takes longer and will return larger matrix of spherical
%   harmonics.
%OUTPUTS
%   f_lm - matrix of spherical harmonics: row = degree starting at 0; cols
%   are order from -max(degree) to +max(degree) with 0-order in the
%   central column.

%--------------------------------------------------------------------------

%get n vectors for lebedev directions
[phi, theta, weight, max_order] = fn_lebedev_quadrature(lebedev_quality);
[x,y,z] = sph2cart(phi, pi / 2 - theta, ones(size(theta)));
n = [x,y,z];

[vph, vgr, p] = fn_anisotropic_vel_profile(stiffness_matrix_or_tensor, rho, n);

switch vph_or_vgr
    case 'vph'
        v = sqrt(sum(vph(:,:,mode_no) .^ 2, 2));
    otherwise
        % THIS IS NOT RIGHT - VGR IS VGR VECTOR FOR PHASE VECTOR IN DIRECTION OF N - WE NEED VGR IN DIRECTION OF N
        % BETTER TO DO THIS AS OPTION IN fn_anisotropic_vel_profile - NOT
        % SURE BEST METHOD
        v = sqrt(sum(vgr(:,:,mode_no) .^ 2, 2)); 
end

degree = [0: round(max_order / 2)]; %gives pretty close to minimum error for given number of Lebedev points
order = [-degree(end): degree(end)];

% v_check = zeros(size(v));
f_lm = zeros(length(degree), length(order));
% current_error = zeros(length(degree), 1);
for r = 1:length(degree) %row
    current_degree = degree(r); %current degree
    [sph_harm, current_orders] = fn_spherical_harmonics(phi, theta, current_degree);
    for i = 1: length(current_orders)
        c = find(order == current_orders(i)); %column associated with order
        f_lm(r,c) = sum(v .* conj(squeeze(sph_harm(:, 1, i))) .* weight);% * (4 * pi); %this is the integration
%         v_check = v_check + squeeze(sph_harm(:, 1, i)) .* f_lm(r, c);
    end
    %calculate error at current level
%     current_error(r) = sqrt(sum(abs(v - real(v_check)) .^ 2));
end

end