function F = fn_hankel_transform(f)
global HANKEL_TRANSFORM_KERNEL
if size(f, 1) == 1
    x = f(:);
else
    x = f;
end;
m = size(x, 1);
r = [0:m-1]';
if m ~= size(HANKEL_TRANSFORM_KERNEL, 1)
    HANKEL_TRANSFORM_KERNEL = fn_recompute_kernal(m);
end;

R = [sqrt(0.5) * (r(1) + r(2)) ^ 2 / 4; ...
    r(3:end) .^ 2 - r(1:end-2) .^ 2 + 2 * r(2:end-1) .* (r(3:end) - r(1:end-2)); ...
    r(end) ^ 2 - (r(end) - r(end-1)) ^ 2 /4] * ones(1, size(x, 2));
F = HANKEL_TRANSFORM_KERNEL * (x .* R);
F = reshape(F, size(f)) / m / sqrt(2 * pi);

return;

function hkr = fn_recompute_kernal(m);
k1 = [0:m-1];
k2 = linspace(0, pi, m + 1);
k2 = k2(1:end-1)'; %my version - pi is one point outside interval
% k2 = linspace(0, pi, m)';%as in library I copied
hkr = besselj(0, k2 * k1);
return;