clear; 
close all;

%Plot spherical harmonics and check orthogonality
max_degree = 4;

lebedev_quality = 3;

do_plots = 1;
orthogonality_check = 1;

plot_as = '2d maps';
% plot_as = '3d surfs';
ang_pts = 45;

%--------------------------------------------------------------------------

degree = [0: max_degree];
order = [-degree(end): degree(end)];

phi = linspace(0,2 * pi, ang_pts * 2 + 1);
theta=linspace(0,    pi, ang_pts + 1);

[phi, theta] = meshgrid(phi, theta);

if do_plots
    for real_imag = [1,2]
        figure;
        for r = 1:length(degree) %row
            current_degree = degree(r); %current degree
            [sph_harm, current_orders] = fn_spherical_harmonics(phi, theta, current_degree);
            for i = 1: length(current_orders)
                c = find(order == current_orders(i)); %column associated with order
                subplot(length(degree), length(order), (r-1)*length(order) + c);
                switch real_imag
                    case 1
                        tmp = real(sph_harm(:,:,i));
                    case 2
                        tmp = imag(sph_harm(:,:,i));
                end
                switch plot_as
                    case '2d maps'
                        imagesc(tmp);
                    case '3d surfs'
                        Cm = sign(tmp);
                        [Xm,Ym,Zm] = sph2cart(phi, pi / 2 - theta, abs(tmp));
                        surf(Xm,Ym,Zm,Cm, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
                        axis equal;
                end
                title(sprintf('(%i, %i)', current_degree, current_orders(i)));
            end
        end
        switch real_imag
            case 1
                set(gcf, 'Name', 'Real');
            case 2
                set(gcf, 'Name', 'Imag');
        end
    end
end

if orthogonality_check
    [phi, theta, weight] = fn_lebedev_quadrature(lebedev_quality);
    all_harms = [];
    harm_labels = {};
    for r = 1:length(degree) %row
        current_degree = degree(r); %current degree
        [sph_harm, current_orders] = fn_spherical_harmonics(phi, theta, current_degree);
        for i = 1: length(current_orders)
            harm_labels{end+1} = sprintf('(%i, %i)', current_degree, current_orders(i));
            all_harms = [all_harms, squeeze(sph_harm(:,1,i))];
        end
    end
    res = zeros(size(all_harms, 2));
    for i = 1:size(all_harms, 2)
        for j = 1:size(all_harms, 2)
            res(i, j) = 4 * pi * sum(all_harms(:,i) .* conj(all_harms(:,j)) .* weight);
        end
    end
    figure;
    imagesc(20*log10(abs(res)));
    caxis([-60, 0]);
    colorbar;
    
    set(gca, 'XTick', [1:size(all_harms, 2)], 'XTickLabel', harm_labels);
    set(gca, 'YTick', [1:size(all_harms, 2)], 'YTickLabel', harm_labels);
    title('Result of integrating pairs of harmonics over unit sphere');
end