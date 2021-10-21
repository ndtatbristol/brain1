function [long_vel, shear_vel] = fn_velocities_from_stiffness_and_density(youngs_modulus, poissons_ratio, density)
%SUMMARY
%   Converts Young's modulus, Poisson's ratio and density to bulk wave
%   velocities
%USAGE
%   [long_vel, shear_vel] = fn_velocities_from_stiffness_and_density(youngs_modulus, poissons_ratio, density)
%INPUTS
%   youngs_modulus - Young's modulus
%   poissons_ratio - Poisson's ratio
%   density - density
%OUTPUTS
%   long_vel - bulk longitudinal wave velocity
%   shear_vel - bulk shear wave velocity
%AUTHOR
%   Paul Wilcox (2007)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

long_vel = sqrt(youngs_modulus .* (1 - poissons_ratio) ./ density ./ (1 + poissons_ratio) ./ (1 - 2 * poissons_ratio));
shear_vel = sqrt(youngs_modulus ./ 2 ./ density ./ (1 + poissons_ratio));

return;