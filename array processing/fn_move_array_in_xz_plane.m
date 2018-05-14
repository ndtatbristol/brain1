function array = fn_move_array_in_xz_plane(old_array, rotation_in_xz_plane, translation_in_xz_plane)
%SUMMARY
%   Transforms array (x,z) coordinates assuming array is data structure in
%   usual NDT lab format
%INPUTS
%   old_array = input array details 
%   rotation_in_xz_plane = angle (+ve = anticlockwise) of rotation in rads
%   translation_in_xz_plane = 2 element vector specifying x and z
%   translation, applied after rotation
%OUTPUTS
%   array = new array details 
%
%--------------------------------------------------------------------------
array = old_array;

translation_in_xz_plane = [translation_in_xz_plane(1), 0, translation_in_xz_plane(2)];

%rotation matrix
m = [cos(rotation_in_xz_plane), 0, sin(rotation_in_xz_plane);0, 1, 0; -sin(rotation_in_xz_plane), 0, cos(rotation_in_xz_plane)];

field_names = regexpi(fieldnames(array), 'el_[xyz][c12]', 'match');

dir = 'xyz';
suffix = 'c12';
% keyboard
old_coords = zeros(length(dir),length(array.el_xc));
for ii = 1:length(suffix)
    for jj = 1:length(dir)
        old_coords(jj,:) = getfield(old_array, ['el_', dir(jj), suffix(ii)]);
    end
    new_coords = m * old_coords;
    for jj = 1:length(dir)
        array.(['el_', dir(jj), suffix(ii)]) = new_coords(jj,:) + translation_in_xz_plane(jj);
    end
end
end