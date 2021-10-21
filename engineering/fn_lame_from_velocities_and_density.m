function [lambda, mu] = fn_lame_from_velocities_and_density(long_vel, shear_vel, density);
%SUMMARY
%   Converts bulk wave velocities and density to Lame constants
%USAGE
%   [lambda, mu] = fn_lame_from_velocities_and_density(long_vel, shear_vel, density)
%INPUTS
%   long_vel - bulk longitudinal wave velocity
%   shear_vel - bulk shear wave velocity
%   density - density
%OUTPUTS
%   lambda - Lame constant lambda
%   mu - Lame constant mu
%AUTHOR
%   Paul Wilcox (2007)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda = density .* (long_vel .^ 2 - 2 * shear_vel .^ 2);
mu = density .* shear_vel .^ 2;

return;


