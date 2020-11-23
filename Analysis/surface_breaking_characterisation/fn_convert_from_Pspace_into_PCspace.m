function PC_point = fn_convert_from_Pspace_into_PCspace(test_par_point,par_points,Bmatrix_pspace,T_surf,PC_points)

%to be consistent with PC_points
par_points = par_points.';

Nnodes = size(T_surf,1);
N_BasisVector = size(par_points,1); 

x_project = zeros(N_BasisVector,Nnodes);

% %reduce number of triangles to check
% %take only closests points
% dist = sum( repmat(test_par_point,1,N_BasisVector) - par_points).^2,2);
% find(dist<=mean(dist)

%find the triangle which contains p_vect
x_nodes = par_points(:,T_surf(:,1));
x_vect = repmat(test_par_point,1,Nnodes) - x_nodes;
ind_nodes = [1:Nnodes];
for jj=1:N_BasisVector
    Bmatrix_jj = squeeze(Bmatrix_pspace(jj,:,ind_nodes));       
    x_project(jj,:) = sum( Bmatrix_jj .* x_vect(:,ind_nodes), 1);
    ind_tmp = find(x_project(jj,:)>=-1e-5);
    ind_nodes = ind_nodes(ind_tmp);
    x_project = x_project(:,ind_tmp);       
end;

PC_point = [];
if length(ind_nodes)
        tmp = sum(x_project,1);
        ind_tmp = find(tmp <= 1 + 1e-5);
        ind_nodes = ind_nodes(ind_tmp);
        x_project = x_project(:,ind_tmp);
        
        if length(ind_nodes)
            ind_el = ind_nodes(1);
            x_project = x_project(:,1);
            PC_surf_el = PC_points(:,T_surf(ind_el,:));
            PC_point = PC_surf_el(:,1) + ( PC_surf_el(:,2:end) - repmat(PC_surf_el(:,1),1,size(T_surf,2)-1) ) * x_project;            
        end;
                       
end;


return;
