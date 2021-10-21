function [v_mean, v_x, v_y, v_z]= fn_get_nominal_velocity(vel_spherical_harmonic_coeffs)
%SUMMARY
%   Returns mean velocity (the degree = 0, order = 0 spherical harmonic coefficient) and 
%   optionally the velocities in the x, y and z directions
v_mean = vel_spherical_harmonic_coeffs(1, (size(vel_spherical_harmonic_coeffs, 2) + 1) / 2);
if nargout > 1
    if prod(size(vel_spherical_harmonic_coeffs)) == 1
        v_x = v_mean;
        v_y = v_mean;
        v_z = v_mean;
    else
        v_x = fn_vel_from_spherical_harmonics(vel_spherical_harmonic_coeffs, 0, pi / 2);
        v_y = fn_vel_from_spherical_harmonics(vel_spherical_harmonic_coeffs, pi / 2, pi / 2);
        v_z = fn_vel_from_spherical_harmonics(vel_spherical_harmonic_coeffs, 0, 0);
    end
end
end