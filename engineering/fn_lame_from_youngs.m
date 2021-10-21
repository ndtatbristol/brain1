function [lambda, mu] = fn_lame_from_youngs(youngs_modulus, poissons_ratio)
%SUMMARY
%   Converts Young's modulus and Poisson's ratio to Lame constants
%USAGE
%   [lambda, mu] = fn_lame_from_youngs(youngs_modulus, poisions_ratio)
%INPUTS
%   youngs_modulus - Young's modulus
%   poissons_ratio - Poisson's ratio
%OUTPUTS
%   lambda - Lame constant lambda
%   mu - Lame constant mu
%AUTHOR
%   Paul Wilcox (2007)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda = (youngs_modulus .* poissons_ratio) ./ ((1 + poissons_ratio) .* (1 - 2 * poissons_ratio));
mu = youngs_modulus ./ (2 * (1 + poissons_ratio));

return;
