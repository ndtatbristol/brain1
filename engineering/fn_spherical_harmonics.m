function [sph_harm, m] = fn_spherical_harmonics(phi, theta, l)
%SUMMARY
%   Returns spherical harmonics Y_lm(phi, theta) of degree l and orders -m
%   to m.
%INPUTS
%   phi, theta - spherical coordinates, must be matrices of same size or
%   one must be scalar. Phi is azimuth measured from x axis in xy plane
%   and theta is elevation measured from z axis.
%   l - degree of spherical harmonics
%OUTPUTS
%   sph_harm - spherical harmonic functions where trailing dimension is
%   order m
%   m - spherical harmonic order

%example plotting
% [Xm,Ym,Zm] = sph2cart(phi, pi / 2 - theta, real(sph_harm)); %NB Matlab measures elevation from xy plane up, not z axis down
% surf(Xm,Ym,Zm); %if phi and theta are from meshgrid
% plot3(Xm,Ym,Zm,'r.'); %if phi and theta are just point clouds


%--------------------------------------------------------------------------

if ~isscalar(phi) && ~isscalar(theta)
    if all(size(phi) == size(theta))
        sz = size(theta);
    else
        error('Inputs theta and phi must be same size or one must be scalar')
    end
else
    if isscalar(theta)
        sz = size(phi);
    else
        sz = size(theta);
    end
end

phi = phi(:);
theta = theta(:);

[p, m] = fn_legendre(l, cos(theta));
sph_harm = zeros(max(length(phi), length(theta)), length(m));
for i = 1 : length(m)
    sph_harm(:, i) = exp(1i * m(i) * phi) .* p(i, :).';
end
sph_harm = reshape(sph_harm, [sz, length(m)]) * sqrt(4*pi);

end

function [p, m] = fn_legendre(l, x) %modification of Matlab one with normalisation and negative orders
sz = size(x);
x = x(:);
p = legendre(l, x, 'norm') / sqrt(2 * pi); %+ve orders
m = [0: size(p, 1) - 1]';
pn = flipud(p(2:end, :)); %-ve orders
pp = p .* (-1) .^ repmat(m, [1, size(p,2)]); %extra sign correction on +ve orders
p = [pn; pp]; %combine
m = [flipud(-m(2:end)); m];
p = reshape(p, [size(p,1), sz]);
end

