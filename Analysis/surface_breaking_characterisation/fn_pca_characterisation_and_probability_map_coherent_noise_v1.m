%general model of coherent noise
function [ par_test, PC_test_project_dist, prob_map, par_prob_map, par_ind_prob_map, Tpar, dpar, PC_test_project_vect  ] = fn_pca_characterisation_and_probability_map_coherent_noise_v1( PC_test, ...
    PC_database, ...
    T_database, PC_BasisVectors_database, PC_Bmatrix_database, Pspace_Bmatrix_database, ...
    par_database, ...
    norm_coef_S_database, mean_S_database, V,...
    noise_std, N_noise, delta_pc, phi_database, ind_S_phi_database, dim_defect_size, ...
    par_prob_map, par_ind_prob_map,...
    threshold_pc)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


Npc_test = length(PC_test);
Ns = size(PC_database,1);
Npc_database = size(PC_BasisVectors_database,1);

Npc = max([Npc_test Npc_database]);

Npar = length(par_database.par_surf);
N_pts_surf = size(PC_database,2);
par_grid_database_norm = zeros(N_pts_surf,Npar);
par_grid_database = zeros(N_pts_surf,Npar);
for ii=1:Npar
    par_grid_database_norm(:,ii) = par_database.par_surf_norm{ii}(par_database.par_ind_surf(:,ii));
    par_grid_database(:,ii) = par_database.par_surf{ii}(par_database.par_ind_surf(:,ii));
end;

%----------------------------------------------------------------------
PC_test = [PC_test; zeros(Ns-Npc_test,1)];

%projection point
[par_test, PC_test_nodes, PC_test_project_dist, PC_test_project_vect]  = fn_pca_characterisation_single_point_v1(PC_test(1:Npc_database), ...
                PC_database(1:Npc_database,:), T_database, PC_BasisVectors_database, PC_Bmatrix_database, par_grid_database ); 
if Npc_test > Npc_database
   PC_test_project_dist = PC_test_project_dist + norm(PC_test(Npc_database+1:Npc_test),2);
end;            
            
%----------------------------------------------------------------------            
% %add projection point to the probability grid
dim_par = zeros(Npar,1);
ind_test_point = zeros(Npar,1);
for ii=1:Npar
    par_min = min(par_database.par_surf{ii});
    par_max = max(par_database.par_surf{ii}); 
    par_test_norm(ii) = (par_test(ii)-par_min)/(par_max-par_min);
    [loc(ii),ind_test_point(ii)] = ismember(par_test_norm(ii), par_prob_map{ii});
    if ~loc(ii)
        ind_test_point(ii) = length(par_prob_map{ii})+1;
        par_prob_map{ii} = sort([par_prob_map{ii}; par_test_norm(ii)]); 
    end;      
    dim_par(ii) = length(par_prob_map{ii});    
end;

[ par_ind_prob_map, Tpar, ~, ~ ] = fn_ngrid_triangulation_v1( dim_par );

%----------------------------------------------------------------------

%generate noise at the projection point
corr_length = [30 ; 10]; %correlation lengths in \theta_1 and \theta_2 directions
Np = 61;
phi_noise = linspace(-90,90,Np)';

S_noise_coh_pc = [];
M=1;

for ii=1:M
[S_noise_coh1,S_noise_coh_pc1] = fn_model_coherent_noise_real_v1(noise_std,corr_length,N_noise,phi_noise,phi_noise,phi_database,V,ind_S_phi_database);
S_noise_coh_pc = [S_noise_coh_pc,S_noise_coh_pc1];
end;
N_noise = N_noise*M;

%--------------------------------------------------------------------------
%probability map
%--------------------------------------------------------------------------
Nnodes = size(par_ind_prob_map,1);
Npar = size(par_ind_prob_map,2);

par_grid_prob_map = zeros(Nnodes,Npar);
for ii=1:Npar
    par_grid_prob_map(:,ii) = par_prob_map{ii}(par_ind_prob_map(:,ii));
end;

%convert noise to PC_noise
noise_mean = mean(S_noise_coh_pc')';
S_noise_coh_pc_cent = S_noise_coh_pc - repmat(noise_mean,1,N_noise);
cov_S = S_noise_coh_pc_cent*S_noise_coh_pc_cent' / (N_noise-1);
[Vn,Dn] = eig(cov_S);
PCn = diag(Dn);
[PCn,ind] = sort(PCn,'descend'); Vn = Vn(:,ind);
PC_Sn = Vn' * S_noise_coh_pc_cent;
noise_amp = real(1./sqrt(2*pi*PCn));
ind_PCn = find(PCn/max(PCn)>=threshold_pc); 

Ncoh = max(ind_PCn);
noise_amp(Ncoh+1:end)=0;

%------------------------------------------

prob_map = zeros(Nnodes,1);
h = waitbar(0,'Probability density calculation ...');
for ii=1:Nnodes    
   
    par_ii = par_grid_prob_map(ii,:).';
    PC_from_par = fn_convert_from_Pspace_into_PCspace(par_ii,par_grid_database_norm,Pspace_Bmatrix_database,T_database,PC_database);
    PC_from_par(Npc_database+1:end) = 0;
    
    if length(PC_from_par)

        PC_noise = PC_test - PC_from_par;
        xx(ii) = norm(PC_noise,2);
        x1 = PC_noise - noise_mean;
        x2 = Vn' * x1;
        Nx2 = Ncoh;                             
        rho = exp(-x2(1:Nx2).^2./PCn(1:Nx2)).*noise_amp(1:Nx2);
        prob_map(ii) = prod(rho);
        
    end;
    waitbar(ii/Nnodes);    
end;
close(h);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NT = size(Tpar,1);
%calculate surface elements
dpar = zeros(NT,1);
for nn=1:NT
    g = par_grid_prob_map(Tpar(nn,[2:end]),:).';
    g0 = par_grid_prob_map(Tpar(nn,1),:).'; 
    g_el = g - repmat(g0,1,Npar);
    dpar(nn) = abs(1/factorial(Npar)*det(g_el));    
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

return;


%**************************************************************************
%**************************************************************************


