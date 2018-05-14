function [ind_surf_el, x_project_surf_el, x_dist_surf] = fn_projection_onto_surface(x, x_surf, T_surf, BasisVectors_surf, Bmatrix_surf)


%find closest point
x_dist = sqrt( sum((repmat(x, 1, size(x_surf,2)) - x_surf).^2, 1) ).';
[x_dist_surf_nearest, ind_min_nearest] = min(x_dist);


%--------------------------------------------------------------------------
%find normal projection
%--------------------------------------------------------------------------

x_nodes = x_surf(:,T_surf(:,1)); 

Nnodes = size(x_nodes,2);
N_BasisVector = size(T_surf,2)-1;

x_vect = repmat(x,1,Nnodes) - x_nodes;
x_project = zeros(N_BasisVector,Nnodes);
    
ind_nodes = [1:Nnodes];
for jj=1:N_BasisVector
    Bmatrix_jj = Bmatrix_surf(jj,:,ind_nodes);
    Bmatrix_jj = permute(Bmatrix_jj,[2 3 1]);
    Bmatrix_jj = squeeze(Bmatrix_jj);
%     Bmatrix_jj = squeeze(Bmatrix_surf(jj,:,ind_nodes));
    
    x_project(jj,:) = sum( Bmatrix_jj .* x_vect(:,ind_nodes), 1);
    ind_tmp = find(x_project(jj,:)>=-1e-5);
    ind_nodes = ind_nodes(ind_tmp);
    x_project = x_project(:,ind_tmp);       
end;

if length(ind_nodes)
        tmp = sum(x_project,1);
        ind_tmp = find(tmp <= 1 + 1e-5);
        ind_nodes = ind_nodes(ind_tmp);
        x_project = x_project(:,ind_tmp);
end;

if length(ind_nodes>=1)
    x_dist = zeros(length(ind_nodes),1);
    for kk=1:length(ind_nodes)
        ind_nodes_kk = ind_nodes(kk);            
        BasisVectors = squeeze(BasisVectors_surf(:,:,ind_nodes_kk)); 
        vect_tmp = x_nodes(:,ind_nodes_kk) + BasisVectors * x_project(:,kk);
        x_dist(kk) = sqrt(sum((x - vect_tmp).^2,1));
    end; 
    [~,ind_min] = min(x_dist);
    ind_surf_el = ind_nodes(ind_min);
    x_project_surf_el = x_project(:,ind_min);    
    x_dist_surf_normal = min(x_dist);
    
    %compare with the distance to nearest point
    if x_dist_surf_normal < x_dist_surf_nearest
        x_dist_surf = x_dist_surf_normal;
    else
        ind_nodes = [];        
    end;
end;
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%if the normal projection doesn't exist or normal distance is greater than the distance to the nearest point 
%--------------------------------------------------------------------------

if ~length(ind_nodes)
    %find all surface elements with the nearest point as a node
    [ind_T_all,ind_n_all] = find(T_surf == ind_min_nearest);
    N_surf_el = length(ind_T_all);
    
    %find the distance to the boundary of each surface element
    ind_surf_el_all = zeros(N_surf_el,1);
    x_project_surf_el_all = zeros(size(T_surf,2)-1,N_surf_el);
    x_dist_surf_all = zeros(N_surf_el,1);
    
    for hh = 1:N_surf_el
        
        ind_T = ind_T_all(hh);
        ind_n = ind_n_all(hh);
        
        if size(T_surf,2) == 2
            ind_surf_el_all(hh) = ind_T;
            x_project_surf_el_all(hh) = ind_n - 1;
            x_dist_surf_all(hh) = x_dist_surf_nearest;
        end;
        
        if size(T_surf,2) > 2
            %create new surface 
            %new points (points of the triangle ind_T)
             x_surf1 = x_surf(:,T_surf(ind_T,:));
            %new triangulation (edges of triangle ind_T)
             N = size(T_surf,2);
             T_surf1 = zeros(N, N-1);
             ind = [1:N]';
             T_surf1(1,:) = ind(2:end);
             ind1 = [ind(2:end);ind(2:2+N-4)];
             for jj=2:size(T_surf,2)
                 T_surf1(jj,:) = [1;ind1(jj-1:jj+N-4)]';
             end;
            %new basis vectors
             BasisVectors_surf1 = zeros(size(x,1), size(T_surf1,2)-1, size(T_surf1,1));
             for jj=1:size(T_surf1,1)
                 for kk=1:size(T_surf1,2)-1
                     BasisVectors_tmp(:,kk) = x_surf1(:,T_surf1(jj,kk+1)) - x_surf1(:,T_surf1(jj,1));
                 end;      
                 BasisVectors_surf1(:,:,jj) = BasisVectors_tmp;
                 G1 = BasisVectors_tmp.' * BasisVectors_tmp;
                 Bmatrix_surf1(:,:,jj) = inv(G1) * BasisVectors_tmp.';
             end;
             [ind_surf_el1, x_project_surf_el1, x_dist_surf1] = fn_projection_onto_surface(x, x_surf1, T_surf1, BasisVectors_surf1, Bmatrix_surf1);
             ind_surf_el_all(hh) = ind_T;
             x_dist_surf_all(hh) = x_dist_surf1;
            %convert projection point into global cordinate system
             BasisVectors = squeeze(BasisVectors_surf1(:,:,ind_surf_el1));
             x_project_global = x_surf1(:,T_surf1(ind_surf_el1,1)) + BasisVectors * x_project_surf_el1; 
            %calculate corrdinate in ind_T element coordinates
             Bmatrix = squeeze(Bmatrix_surf(:,:,ind_T));
             x_project_surf_el_all(:,hh) = Bmatrix * (x_project_global-x_nodes(:,ind_T));    
        end;
        
    end;%for hh = 1:N_surf_el  
    
 [~,ind_min] = min(x_dist_surf_all);
 ind_surf_el = ind_surf_el_all(ind_min);
 x_project_surf_el = x_project_surf_el_all(:,ind_min);
 x_dist_surf = x_dist_surf_all(ind_min);
    
end;%~length(ind_nodes)



return;