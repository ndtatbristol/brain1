function [d, n_hat] = fn_find_flat_surf_from_pe_data(exp_data, filter, couplant_velocity, min_dist_to_surf, max_dist_to_surf)
%SUMMARY
%Finds flat surface from pulse-echo part of FMC data in immersion test.
%INPUTS
%exp_data - experimental data in usual form
%filter - either  [] or 0 for no filter, otherwise a freq domain filter
%function of size(filter) = size(exp_data.time)
%couplant_velocity - couplant velocity
%min_dist_to_surf - minimum distance from any array element to surface
%to consider
%max_dist_to_surf - maximum distance from any array element to surface to
%consider
%OUTPUTS
%d - distance from array cenntre to point on surface in direction of
%array normal (i.e. along the z axis in array coordinates)
%n_hat - 3x1 vector of the unit normal of the surface.


pe_data = exp_data.time_data(:, find(exp_data.tx == exp_data.rx));
if isempty(filter) || ~(length(filter) == length(exp_data.time))
    pe_data = abs(fn_hilbert(pe_data));
else
    pe_data = abs(ifft(spdiags(filter, 0, length(exp_data.time), length(exp_data.time)) * fft(pe_data)));
end

%window in time-domain
t1 = min_dist_to_surf * 2 / couplant_velocity;
t2 = max_dist_to_surf * 2 / couplant_velocity;
pe_data = pe_data .* ((exp_data.time >= t1 & exp_data.time <= t2) * ones(1, size(pe_data,2)));
% keyboard
%find arrival times of largest signals (assumed to be surface reflections)
[dummy, ii] = max(pe_data);
t = exp_data.time(ii);

%fit linear function to pe data points and calculate n and d
switch fn_return_dimension_of_array(exp_data.array)
    case 1
        x_hat = [1,0,0];
        z_hat = [0,0,1];
        p = polyfit(exp_data.array.el_xc(:), t(:) * couplant_velocity / 2, 1);
        n_hat = [p(1), 0, -1]; %error on this line! should be n_hat = [p(1), 0, -sqrt(1-p(1)^2))];
        n_hat = n_hat / sqrt(sum(n_hat .^ 2));
        a = n_hat * polyval(p, 0);
        b = [1, 0, p(1)];
        lambda = - dot(x_hat, a) / dot(x_hat, b);
        p = a + lambda * b;
        d = abs(p(3));
    case 2
        %not implemented yet - need to use SVD

end
end