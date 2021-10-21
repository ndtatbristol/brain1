function f_lm = fn_spherical_harmonics_from_velocity(v, max_order_l, varargin)
%SUMMARY
%   Converts velocity vectors, v, in different directions into spherical
%   harmonics up to specified order
%INPUTS
%   v - [n x 3] matrix of velocity components in x, y and z
%   max_order_l - maximum order of spherical harmonic in elevation
%   [max_order_m] - maximum order of spherical harmonic in azimuth,
%   defaults to max allowed for elevation order but can be less
%NOTES
%   The velocities can be in absolutely any order, but there is no checking
%   to ensure complete coverage of angles.

%--------------------------------------------------------------------------
if isempty(varargin)
    max_order_m = max_order_l;
else
    max_order_m = min(varargin{1}, max_order_l);
end

ang_pts = max_order_l * 4; %no idea whether this is good factor or not!
phi = linspace(0, 2 * pi, ang_pts + 1)'; %azimuthal angle
theta = linspace(0, pi, ang_pts / 2 + 1)'; %elevation angle
phi = phi(1:end-1);
% theta = theta(2:end-1);
[PHI, THETA] = meshgrid(phi, theta);
dphi = phi(2) - phi(1);
dtheta = theta(2) - theta(1);

[phi1, theta1, v1] = cart2sph(v(:, 1), v(:, 2), v(:, 3));
%correct angle definitions
phi1 = phi1 + pi;
theta1 = pi / 2 - theta1;
%tile in both angles to avoid problems at angle limits (what a
%pain!)
v2 = [];
phi2 = [];
theta2 = [];
for phi_step = [-2*pi, 0, 2*pi]
    for theta_step = [-pi, 0, pi]
        v2 = [v2; v1];
        phi2 = [phi2; phi1 + phi_step];
        theta2 = [theta2; theta1 + theta_step];
    end
end
F = scatteredInterpolant(phi2, theta2, v2, 'linear', 'none');
v = F(PHI, THETA);

%Now we have v sampled uniformly over phi = [0, 2*pi] and theta = [0, pi]
%we need to do the transformation to spherical harmonics

%row, l, and col, m, indices for spherical harmonics
l = [0: size(v, 1)];
m = [-l(end): l(end)];

sz = size(v);
PHI = PHI(:);
THETA = THETA(:);
v = v(:);
v2 = zeros(size(v));
f_lm = zeros(length(l), length(m));
for r = 1:length(l) %row
    ll = l(r); %current degree
    [sph_harm, mm] = fn_spherical_harmonics(PHI, THETA, ll);
    for i = 1: length(mm)
        c = find(m == mm(i)); %column associated with order
        f_lm(r,c) = sum(v .* conj(sph_harm(:, i)) .* sin(THETA)) * dphi * dtheta; %integral could be better
        v2 = v2 + sph_harm(:, i) .* f_lm(r,c);
    end
end

v2 = reshape(v2, sz);

end

