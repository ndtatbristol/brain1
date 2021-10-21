function D = fn_iso_stiffness_matrix(E, v)
%SUMMARY
%   Converts Young's modulus and Poisson's ratio into 6x6 stiffness matrix
%USAGE
%   D = fn_iso_stiffness_matrix(E, v)
%INPUTS
%   E - Young's modulus (scalar)
%   v - Poisson's ratio (scalar)
%OUTPUTS
%   D - 6x6 stiffness matrix
%AUTHOR
%   Paul Wilcox (2010)
%--------------------------------------------------------------------------

D = E / (1 + v) / (1 - 2 * v) * [...
    [1-v,   v,      v,      0,              0,              0];...
    [v,     1-v,    v,      0,              0,              0];...
    [v,     v,      1-v,    0,              0,              0];...
    [0,     0,      0,      0.5*(1-2*v),    0,              0];...
    [0,     0,      0,      0,              0.5*(1-2*v),    0];...
    [0,     0,      0,      0,              0,              0.5*(1-2*v)]];
return