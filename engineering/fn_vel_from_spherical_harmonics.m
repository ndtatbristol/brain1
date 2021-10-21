function v = fn_vel_from_spherical_harmonics(f_lm, phi, theta)
%SUMMARY
%   Computes velocity at given spherical angle (phi, theta) using specified
%   spherical harmonic coefficients.
%INPUTS
%   f_lm - spherical harmonic coefficients. Must have odd number of columns
%   as these are interpreted as orders from -max_order to max_order with
%   central column as order 0.
%   phi - azimuthal angle (relative to x-axis in x-y plane)
%   theta - elevation angle (0 to pi, measured from z-axis)
%NOTES
%   Follows standard spherical harmonic notation on Wikipedia
%   (https://en.wikipedia.org/wiki/Spherical_harmonics)
%--------------------------------------------------------------------------

if rem(size(f_lm,2), 2) ~= 1
    error('f_lm matrix must have odd number of columns');
end

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

degree = [0: size(f_lm, 1) - 1];
max_order = round((size(f_lm, 2) - 1) / 2);
order = [-max_order: max_order];

%set all harmonics below threshold to zero to avoid calculating these
f_lm(abs(f_lm) / max(abs(f_lm(:))) < 0.0001) = 0;


%loop through the harmonics and accumulate the contributions in v
v = zeros(size(phi));
for r = 1:length(degree) %row
    if any(f_lm(r, :)) %only calculate if nesc
        current_degree = degree(r); %current degree
        [sph_harm, current_order] = fn_spherical_harmonics(phi, theta, current_degree);
        for i = 1: length(current_order)
            c = find(order == current_order(i)); %column associated with order
            if ~isempty(c)
                v = v + sph_harm(:, i) * f_lm(r, c);
            end
        end
    end
end

v = real(reshape(v, sz));

end