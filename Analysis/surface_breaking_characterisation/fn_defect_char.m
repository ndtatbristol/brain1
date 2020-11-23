function [prob_map_plot1, T_par_crack, p_par_crack, par_test1]=fn_defect_char(s, exp_data, options)
%Characterisation of surface-breaking cracks using Gaussian coherent noise
%model

fl_sub_array = 0;
num_el = length(exp_data.array.el_xc);
num_sub_el = options.aperture_els;

PC_threshold=0.005;
PC_threshold_test=0.005*2;
fl_include_phase=0;
fl_shape_only=0;
delta_pc = 1;
noise_std = 0.1;

%probability map
dl_prob_crack = 0.1; %wavelength
da_prob_crack = 5; %degrees

s.m = abs(s.m);
s.phi = - s.phi; %different angle convention is used here

%filtering
fmatrix = fft2(s.m); 
fmatrix = fftshift(fmatrix);  
[dummy, ii] = max(abs(fmatrix));
[dummy, jj] = max(dummy);
ii = ii(jj); 
k = [(1-ii):(size(s.phi,2)-ii)]';
[k1,k2] = meshgrid(k,k); 
R=15; 
threshold=0.01;
b = log(1/threshold);
filter = exp(-(k1.^2+k2.^2)/R^2 * b);
fmatrix_filt = fmatrix .* filter;
fmatrix_filt = ifftshift(fmatrix_filt);
s.m = ifft2(fmatrix_filt);
%scaling
gain = 10; %gain used to scale up experimental S-matrix (for sub-array aperture size of 8)
s.m = s.m * gain;
s.m = s.m(4:53, 4:53);
s.phi = s.phi(4:53);

phi_exp = s.phi*180/pi;

%full set of angles
phi_in_database = phi_exp;
phi_sc_database = phi_exp;
[pd1,pd2] = ndgrid(phi_sc_database*pi/180,phi_in_database*pi/180);
ind_S_phi = find(pd1>=pd2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%surface cracks%%% parameters: 1. length   2. angle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
crack_length_min = 0.5; crack_length_max = 2;
crack_angle_min = -60; crack_angle_max = 60;
%--------------------------------------------------------------------------



%--------------------------------------------------------------------------
%create database of cracks
path = '';
fname = 'S_surface_crack_L_05_3_Ang_0_60.mat';
path_crack_database = [path,fname];
lim_crack_length = [crack_length_min crack_length_max];
lim_crack_angle = [crack_angle_min crack_angle_max];
[S_crack_database, par_crack_database] = fn_prepare_global_database_surface_crack_single_freq(path_crack_database,lim_crack_length,lim_crack_angle);

if ~fl_sub_array
[ S_database_complex1, PC_database1, norm_coef_S_database1, mean_S_database1, V1, PC1, ind_PC1,...
    T_surf1, Bmatrix_surf1, BasisVectors_surf1, Bmatrix_pspace1] = ...
    fn_prepare_database_v1( S_crack_database, par_crack_database, phi_in_database*pi/180, phi_sc_database*pi/180, ind_S_phi, PC_threshold, fl_include_phase, fl_shape_only);
else
[ S_database_complex1, PC_database1, norm_coef_S_database1, mean_S_database1, V1, PC1, ind_PC1,...
    T_surf1, Bmatrix_surf1, BasisVectors_surf1, Bmatrix_pspace1] = ...
    fn_prepare_database_sub_array( S_crack_database, par_crack_database, phi_in_database*pi/180, phi_sc_database*pi/180, ind_S_phi, PC_threshold, ...
    fl_include_phase, fl_shape_only, num_el, num_sub_el);
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S_test = s.m(ind_S_phi); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%crack database
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%characterisation parameters
delta_pc1 = delta_pc;
N_noise1 = 1000;
noise_std1 = noise_std;
dim_defect_size1 = 1;
%probability map
Nl_prob = fix((crack_length_max-crack_length_min)/dl_prob_crack)+1;
Na_prob = fix((crack_angle_max-crack_angle_min)/da_prob_crack)+1;
N_crack = [Nl_prob Na_prob];

for ii=1:2
   par_prob_map_norm_crack{ii} = linspace(0,1,N_crack(ii))';
end;
[p1,p2] = ndgrid([1:N_crack(1)],[1:N_crack(2)]);
par_ind_prob_map_crack = [p1(:),p2(:)];
Npar_crack = 2;
for ii=1:Npar_crack
    par_min = min(par_crack_database.par_surf{ii});
    par_max = max(par_crack_database.par_surf{ii}); 
    par_prob_map_crack{ii} = par_prob_map_norm_crack{ii}*(par_max-par_min) + par_min;
end;

PC_test1 = fn_convert_Smatrix_into_PC_space( S_test, norm_coef_S_database1, mean_S_database1, V1);
ind = find(abs(PC_test1)/max(abs(PC_test1))>=sqrt(PC_threshold)); Nx = max(ind);
[ par_test1, PC_test_project_dist1, prob_map1, par_prob_map_norm_crack1, par_ind_prob_map_crack1, T_par_crack, dp_crack, PC_test_project_vect ] = ...
    fn_pca_characterisation_and_probability_map_coherent_noise_v1( PC_test1(1:Nx), ...
    PC_database1, ...
    T_surf1, BasisVectors_surf1, Bmatrix_surf1, Bmatrix_pspace1, ...
    par_crack_database, ...
    norm_coef_S_database1, mean_S_database1, V1,...
    noise_std1, N_noise1, delta_pc1, phi_exp, ind_S_phi, dim_defect_size1, ...
    par_prob_map_norm_crack, par_ind_prob_map_crack, PC_threshold);

%update probability grid
p_par_crack = zeros(size(par_ind_prob_map_crack1,1), Npar_crack);
for ii=1:Npar_crack
    par_min = min(par_crack_database.par_surf{ii});
    par_max = max(par_crack_database.par_surf{ii}); 
    par_prob_map_crack1{ii} = par_prob_map_norm_crack1{ii}*(par_max-par_min) + par_min;
    p_par_crack(:,ii) = par_prob_map_crack1{ii}(par_ind_prob_map_crack1(:,ii));
    N_crack(ii) = length(par_prob_map_crack1{ii});
end;

prob_map_plot1 = prob_map1;

% figure(1);
% patch('Faces',T_par_crack,'Vertices',p_par_crack,'FaceColor','interp','FaceVertexCData',(prob_map_plot1)/max(abs(prob_map_plot1(:))),'LineWidth',0.1,'LineStyle','none');
% colorbar;
% ylabel('orientation angle (\circ)', 'FontSize', 18)
% xlabel('size (\lambda)', 'FontSize', 18)
% set(gca,'fontsize',16)
% hold on
% plot(par_test1(1), par_test1(2), 'ro', 'LineWidth', 2)

if options.display_manifold
    %%plot manifold and projection
    p1 = -60:2:60; p2 = 0.5:0.1:2;
    plot_surf_2d(PC_database1, p1, p2);
    hold on
    plot3(PC_test1(1), PC_test1(2), PC_test1(3), 'bo', 'MarkerFaceColor', 'b')
    plot3(PC_test_project_vect(1), PC_test_project_vect(2), PC_test_project_vect(3), 'ro', 'MarkerFaceColor', 'r')
end













