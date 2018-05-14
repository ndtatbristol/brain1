function [ S_database_complex, PC_database, norm_coef_S_database, mean_S_database, V, PC, ind_PC,...
    T_surf, Bmatrix_surf, BasisVectors_surf, Bmatrix_pspace] = ...
    fn_prepare_database_v1( S_database_global, par_database, phi_in, phi_sc, ind_S_phi, PC_threshold, fl_include_phase, fl_shape_only)

phi_in_database_global = S_database_global.phi_in;
phi_sc_database_global = S_database_global.phi_sc;
% %freq_database_global = S_database_global.freq;
% % [x1,x2,x3] = ndgrid(phi_database_global,phi_database_global,freq_database_global);
% [y1,y2,y3] = ndgrid(phi_sc,phi_in,freq);

[x1,x2] = ndgrid(phi_sc_database_global,phi_in_database_global);
[y1,y2] = ndgrid(phi_sc,phi_in);
ind_y12 = ind_S_phi;%find(y1>=y2);

NS_database = length(S_database_global.S);
NL_database = length(ind_y12(:));
if fl_include_phase
    NL_database = NL_database*2;
end;
S_database = zeros(NL_database,NS_database);
S_database_complex = S_database;

h = waitbar(0,'Preparing database ...');
for ii=1:NS_database
    
    S_global_tmp = S_database_global.S{ii};
%     freq_database_global = S_database_global.freq{ii};
%     [x1,x2,x3] = ndgrid(phi_sc_database_global,phi_in_database_global,freq_database_global);    
%     S_tmp = interpn(x1,x2,x3,S_global_tmp,y1,y2,y3,'spline');
    S_tmp = interpn(x1,x2,S_global_tmp,y1,y2,'spline',0);
    S_tmp = S_tmp(ind_y12);
    S_tmp = S_tmp(:);
    S_database_complex(:,ii) = S_tmp;
    if fl_include_phase
        S_tmp = [real(S_tmp); imag(S_tmp)];
    else
        S_tmp = abs(S_tmp);
    end;
    S_database(:,ii) = S_tmp;
    
    
    waitbar(ii/NS_database);
end;
close(h);

%--------------------------------------------------------------------------
if fl_shape_only
    
    S_max = max(S_database,[],1);
    S_database = S_database ./ repmat(S_max,size(S_database,1),1);
    
end;

%--------------------------------------------------------------------------
h = waitbar(1,'Converting database into principal components ...');

%normalisation coefficient
norm_coef_S_database = 1;%max(max(abs(S_database)));
S_database_norm = S_database / norm_coef_S_database;
%subscribe mean
mean_S_database = mean(S_database_norm')';
S_database_cent = S_database_norm - repmat(mean_S_database,1,NS_database);
cov_S = S_database_cent*S_database_cent' / (NS_database-1);
[V,D] = eig(cov_S);
PC = diag(D);
[PC,ind] = sort(PC,'descend'); V = V(:,ind);
PC_database = V' * S_database_cent;

close(h);

%--------------------------------------------------------------------------

h = waitbar(1,'Discretizing P-surface ...');

ind_PC = find(PC/PC(1)>=PC_threshold);

x_surf = PC_database(ind_PC,:);
[T_surf, Bmatrix_surf, BasisVectors_surf, Bmatrix_pspace] = fn_surface_triangulation_v1(par_database.par_surf_norm,par_database.par_ind_surf,x_surf);

close(h);

return;

