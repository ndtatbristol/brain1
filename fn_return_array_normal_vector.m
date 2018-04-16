function n = fn_return_array_normal_vector(array)
%SUMMARY
%   Returns array normal unit vector (3x1) for a 1D or 2D array. If array
%   is non-planar, this will be the mean normal vector for all elements
%INPUT
%   array - standard format array structure
%OUTPUT
%   n - 3x1 unit normal vector
%--------------------------------------------------------------------------
n = zeros(3, length(array.el_xc));
for ii = 1:length(array.el_xc)
    d1 = [array.el_x1(ii) - array.el_xc(ii); array.el_y1(ii) - array.el_yc(ii);  array.el_z1(ii) - array.el_zc(ii)];
    d2 = [array.el_x2(ii) - array.el_xc(ii); array.el_y2(ii) - array.el_yc(ii);  array.el_z2(ii) - array.el_zc(ii)];
    n(:,ii) = cross(d1, d2);
end
n = mean(n, 2);
n = n / sqrt(sum(n .^ 2));
end