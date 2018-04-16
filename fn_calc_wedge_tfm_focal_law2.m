function focal_law = fn_calc_wedge_tfm_focal_law2(exp_data, mesh, wedge_vel, varargin)
%array el positions must have been already moved to correct position with
%respect to mesh - z-origin of mesh is always at surface

method = '1C';
debugging = 0;
pts_per_lambda = 8;

%find out dimension of mesh (i.e. 2D or 3D image to be produced
orig_dim = size(mesh.x);
mesh.x = mesh.x(:);
mesh.z = mesh.z(:);
if isfield(mesh, 'y')
    mesh.y = mesh.y(:);
    n = 3;
else
    n = 2;
end

focal_law.lookup_time = zeros(length(mesh.x), length(exp_data.array.el_xc));
r = wedge_vel / exp_data.ph_velocity;

min_lambda = min([wedge_vel, exp_data.ph_velocity]) / exp_data.array.centre_freq;
x1 = min([exp_data.array.el_xc(:); mesh.x(:)]);
x2 = max([exp_data.array.el_xc(:); mesh.x(:)]);
xx = linspace(x1, x2, (x2 - x1) / min_lambda * pts_per_lambda);

for ti = 1:length(exp_data.array.el_xc) %loop over elements in array
    for mi = 1:length(mesh.x) %loop over image points
        switch method
            case '1A'
                %method 1A - solve quartic
                x = mesh.x(mi) - exp_data.array.el_xc(ti);
                z1 = -exp_data.array.el_zc(ti);
                z2 = mesh.z(mi);
                c(1) = (1-r^2);
                c(2) = -2*x*(1-r^2);
                c(3) = (1-r^2) * x^2 + z2^2 - r^2 * z1^2;
                c(4) = 2*r^2*z1^2*x;
                c(5) = -r^2*z1^2*x^2;
                p = roots(c);
                t = sqrt(z1 ^ 2 + real(p) .^ 2) / wedge_vel + sqrt(z2 ^ 2 + (x - real(p)) .^ 2) / exp_data.ph_velocity;
                [focal_law.lookup_time(mi, ti), ii] = min(t);
                p1 = exp_data.array.el_xc(ti) + p(ii);
                
            case '1B'
                %method 1B - solve minimisation
                x0 = mean([mesh.x(mi), exp_data.array.el_xc(ti)]);
                p2 = fminsearch(@(xx) fn_time_taken(exp_data.array.el_xc(ti), exp_data.array.el_zc(ti), wedge_vel, mesh.x(mi), mesh.z(mi), exp_data.ph_velocity, xx), x0);
                focal_law.lookup_time(mi, ti) = fn_time_taken(exp_data.array.el_xc(ti), exp_data.array.el_zc(ti), wedge_vel, mesh.x(mi), mesh.z(mi), exp_data.ph_velocity, p2);
                if debugging
                    h2 = plot(...
                        [exp_data.array.el_xc(ti), p2, mesh.x(mi)], ...
                        [exp_data.array.el_zc(ti), 0, mesh.z(mi)], 'b:');
                    pause(0.00001);
                    delete(h2);
                end
            case '1C'
                %method 1C - find minima from set of points
                x1 = min([exp_data.array.el_xc(ti); mesh.x(mi)]);
                x2 = max([exp_data.array.el_xc(ti); mesh.x(mi)]);
                xx = linspace(x1, x2, (x2 - x1) / min_lambda * pts_per_lambda);
                t = fn_time_taken(exp_data.array.el_xc(ti), exp_data.array.el_zc(ti), wedge_vel, mesh.x(mi), mesh.z(mi), exp_data.ph_velocity, xx);
%                 keyboard
                [focal_law.lookup_time(mi, ti), ii] = min(t);
                p3 = xx(ii);
                if debugging
                    h3 = plot(...
                        [exp_data.array.el_xc(ti), p3, mesh.x(mi)], ...
                        [exp_data.array.el_zc(ti), 0, mesh.z(mi)], 'b:');
                    pause(0.00001);
                    delete(h3);
                end
        end
        %         %debugging
        %         hold on;
        %         h = plot(...
        %             [exp_data.array.el_xc(ti), p1, mesh.x(mi)], ...
        %             [exp_data.array.el_zc(ti), 0, mesh.z(mi)], 'g');
        %         h2 = plot(...
        %             [exp_data.array.el_xc(ti), p2, mesh.x(mi)], ...
        %             [exp_data.array.el_zc(ti), 0, mesh.z(mi)], 'b:');
        %         pause(0.001);
        % %         keyboard;
        %         delete(h);
        %         delete(h2);
    end
    fn_show_progress(ti, length(exp_data.array.el_xc))
end
end

function t = fn_time_taken(x1, z1, v1, x2, z2, v2, p)
t = sqrt((x1 - p) .^ 2 + z1 ^ 2) / v1 + sqrt((x2 - p) .^ 2 + z2 ^ 2) / v2;
end
