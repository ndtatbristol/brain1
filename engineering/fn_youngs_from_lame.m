function [youngs_modulus, poissons_ratio] = fn_youngs_from_lame(lambda, mu);
%SUMMARY
%   Converts Lame constants to Young's modulus and Poisson's ratio
%USAGE
%   [youngs_modulus, poisions_ratio] = fn_youngs_from_lame(lambda, mu)
%AUTHOR
%   Paul Wilcox (2007)
%INPUTS
%   lambda - Lame constant lambda
%   mu - Lame constant mu
%OUTPUTS
%   youngs_modulus - Young's modulus
%   poissons_ratio - Poisson's ratio

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

youngs_modulus = ((3 * lambda + 2 * mu) .* mu) ./ (lambda + mu);
poissons_ratio = lambda ./ (2 * (lambda + mu));

return;

