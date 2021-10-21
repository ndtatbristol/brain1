function [youngs_modulus, poissons_ratio] = fn_stiffness_from_velocities_and_density(long_vel, shear_vel, density)
%SUMMARY
%   Converts bulk wave velocities and density to Young's modulus and
%   Poisson's ratio
%USAGE
%   [youngs_modulus, poissons_ratio] = fn_stiffness_from_velocities_and_density(long_vel, shear_vel, density)
%INPUTS
%   long_vel - bulk longitudinal wave velocity
%   shear_vel - bulk shear wave velocity
%   density - density
%OUTPUTS
%   youngs_modulus - Young's modulus
%   poissons_ratio - Poisson's ratio
%AUTHOR
%   Paul Wilcox (2007)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[lambda, mu] = fn_lame_from_velocities_and_density(long_vel, shear_vel, density);
[youngs_modulus, poissons_ratio] = fn_youngs_from_lame(lambda, mu);

return;