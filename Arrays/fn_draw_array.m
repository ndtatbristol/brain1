function fn_draw_array(array, line_color, patch_color, swap_xz)
degs_per_seg = 5;
for i = 1:length(array.el_xc)
    [x, y] = fn_get_coords(...
        [array.el_xc(i), array.el_yc(i), array.el_zc(i)], ...
        [array.el_x1(i), array.el_y1(i), array.el_z1(i)], ...
        [array.el_x2(i), array.el_y2(i), array.el_z2(i)], ...
        array.el_type, ...
        degs_per_seg ...
        );
    if swap_xz
        %necessary for 2D views where NDE axes are x and z but Matlab plots are
        %x and y
        z = y;
        y = zeros(size(x));
    else
        z = zeros(size(x));
    end
    if isempty(patch_color)
        line(x, y, z, 'Color', line_color);
    else
        patch(x, y, z, patch_color, 'EdgeColor', line_color);
    end
end

end

function [x, y] = fn_get_coords(p0, p1, p2, t, degs_per_seg)
switch t
    case {'rectangular', 'elliptical'}
        [a1, a2, theta] = fn_axes_and_angle(p0, p1, p2);
        switch t
            case 'rectangular'
                a = linspace(0,2*pi, 5)' + pi / 4; %trick to get corner points using angles
                m = sqrt(2);
            case 'elliptical'
                a = linspace(0,2*pi, 360 / degs_per_seg)';
                m = 1;
        end
        xy = [cos(a), sin(a)] * m;
        [x, y] = fn_rotate_and_shift(xy .* [a1, a2], theta, p0);
    case 'annular'
        r0 = sqrt(sum(p0 .^ 2));
        r1 = sqrt(sum(p1 .^ 2)); %by convention, p1 is the one on same angle, different radius
        r2 = r0 - (r1 - r0);
        if r0 > 0
            t0 = atan2(p0(2), p0(1));
            t2 = atan2(p2(2), p2(1)); %by convention, p2 is the one on same radus, different angle
            t1 = t0 - (t2 - t0);
            a = linspace(t1, t2, ceil(abs(t2 - t1) * 180 / pi / degs_per_seg) + 1); %to change number
            x = [r1 * cos(a), r2 * cos(fliplr(a)), r1 * cos(a(1))];
            y = [r1 * sin(a), r2 * sin(fliplr(a)), r1 * sin(a(1))];
        else
            a = linspace(0, 2*pi, 360 / degs_per_seg + 1);
            x = r1 * cos(a);
            y = r1 * sin(a);
        end
end

end

function [a1, a2, theta] = fn_axes_and_angle(p0, p1, p2)
a1 = sqrt(sum((p1 - p0) .^ 2));
a2 = sqrt(sum((p2 - p0) .^ 2));
theta = atan2(p1(2) - p0(2), p1(1) - p0(1));
end

function [x, y] = fn_rotate_and_shift(xy, theta, p0)
xy = [xy; xy(end, :)];
xy = xy * [cos(theta), sin(theta); -sin(theta), cos(theta)] + p0(1:2);
x = xy(:, 1)';
y = xy(:, 2)';
end