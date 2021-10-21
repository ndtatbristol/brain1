function D = fn_tranversely_isotropic_stiffness_matrix(Ez, Ep, vpz, vpp, Gzp)
%SUMMARY
%   Converts Engineering matl properties into 6x6 stiffness matrix
%USAGE
%   D = fn_tranversely_isotropic_stiffness_matrix(Ez, Ep, vpz, vpp, G)
%INPUTS
%   Ez - Young's modulus in axial direction (scalar)
%   Ep - Young's modulus in in-plane direction (scalar)
%   vpz - Poisson's ratio for in-plane / axial direction (scalar)
%   vpp - Poisson's ratio for in-plane directions (scalar)
%   Gzp - shear modulus in axial direction (scalar)
%OUTPUTS
%   D - 6x6 stiffness matrix
%AUTHOR
%   Paul Wilcox (2012)
%NOTES
%   Equation transcribed from http://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_iso_transverse.cfm
%   for engineering shear strain
%--------------------------------------------------------------------------
vzp = Ez / Ep * vpz;
d = (1 + vpp) * (1 - vpp - 2 * vpz * vzp) / (Ep ^ 2 * Ez)
D = [[(1 - vpz * vzp) / (Ep * Ez * d), (vpp + vpz * vzp) / (Ep * Ez * d), (vzp + vpp * vzp) / (Ep * Ez * d), 0 ,0 ,0]; ...
    [(vpp + vpz * vzp) / (Ep * Ez * d), (1 - vpz * vzp) / (Ep * Ez * d), (vzp + vpp * vzp) / (Ep * Ez * d), 0, 0, 0]; ...
    [(vpz + vpp * vpz) / (Ep ^ 2 * d), vpz * (1 + vpp) / (Ep ^ 2 * d), (1 - vpp ^ 2) / (Ep ^ 2 * d), 0, 0, 0]; ...
    [0, 0, 0, Gzp, 0, 0];
    [0, 0, 0, 0, Gzp, 0];
    [0, 0, 0, 0, 0, Ep / (1 + vpp) / 2]];
end