function point = fn_line_plane_intersection(line, plane)
%INPUTS
%   line - 2 x 3 matrix describing two points on line
%   plane - 3 x 3 matrix describing three points in plane
%OUTPUT
%   point  - 1 x 3 vector of interection point

%matrix to describe problem
m = [line(1,:) - line(2,:); plane(2,:) - plane(1,:); plane(3,:) - plane(1,:)]';
v = [line(1,:) - plane(1,:)]';
r = m \ v;
point = line(1,:) + (line(2,:) - line(1,:)) * r(1);
end