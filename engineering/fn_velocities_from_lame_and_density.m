function [long_vel, shear_vel] = fn_velocities_from_lame_and_density(lambda, mu, density)
%SUMMARY
%   Converts Lame constants and density to bulk wave velocities
%USAGE
%   [long_vel, shear_vel] = fn_velocities_from_lame_and_density(lambda, mu, density)
%INPUTS
%   lambda - Lame constant lambda
%   mu - Lame constant mu
%   density - density
%OUTPUTS
%   long_vel - bulk longitudinal wave velocity
%   shear_vel - bulk shear wave velocity
%AUTHOR
%   Paul Wilcox (2007)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[youngs_modulus, poissons_ratio] = fn_youngs_from_lame(lambda, mu);
[long_vel, shear_vel] = fn_velocities_from_stiffness_and_density(youngs_modulus, poissons_ratio, density);

return;