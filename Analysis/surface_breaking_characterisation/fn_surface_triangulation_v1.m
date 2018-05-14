%triangulate parametrically defined surface in n-dimensional space
%takes into account periodic parameters
function [T_surf, Bmatrix_surf, BasisVectors_surf, Bmatrix_pspace] = fn_surface_triangulation_v1(par_surf_norm, par_ind_surf, x_surf)

dim_x = size(x_surf,1);
N_BasisVector = size(par_ind_surf,2); 

%periodic parameters
% % par_surf1 = par_surf_norm;
% % par_ind_surf1 = par_ind_surf;
% % nn=0;
% % for ii=1:length(par_surf_norm)
% %     if max(par_surf_norm{ii})<1
% %         nn=nn+1;
% %         par_ind_period(nn)=ii;
% %         par_surf1{ii} = [par_surf_norm{ii}; 1];
% %         
% %         ind_add{nn} = find(par_ind_surf1(:,ii)==1);
% %         par_ind_add = par_ind_surf1(ind_add{nn},:);
% %         par_ind_add(:,ii) = length(par_surf1{ii});
% %         
% %         ind_add1{nn} = size(par_ind_surf1,1) + [1:length(ind_add{nn})]';
% %         par_ind_surf1 = [par_ind_surf1; par_ind_add];
% % 
% %     end;
% % end;
% for ii=1:length(par_ind_period)    
%     ind_tmp = find(par_ind_surf(:,par_ind_period(ii))==1);
%     par_ind_tmp = par_ind_surf(ind_tmp,:);
%     par_ind_tmp(:,par_ind_period(ii)) = 0;
%     [loc1,loc2] = ismember(par_ind_tmp,par_ind_surf,'rows');
%     ind_period{ii} = [ind_tmp, loc2];
% end;


x_pts = size(par_ind_surf,1);
par_points = zeros(x_pts,N_BasisVector);
for ii=1:N_BasisVector
    par_points(:,ii) = par_surf_norm{ii}(par_ind_surf(:,ii));
end;

%triangulate new surface
    T_pspace = delaunayn(par_points,{'QJ'});
    Nnodes = size(T_pspace,1);
    T_surf = T_pspace;
    
%     for ii=1:length(par_ind_period) 
%         ind = find(T_pspace==ind_period{ii}(:,1));
%         T_surf(ind) = ind_period{ii}(:,2);               
%     end;
        

    %calculate basis vectors for each surface element
    Bmatrix_surf = zeros(N_BasisVector,dim_x,Nnodes);
    BasisVectors_surf = zeros(dim_x,N_BasisVector,Nnodes);
    Bmatrix_pspace = zeros(N_BasisVector,N_BasisVector,Nnodes);
    for nn=1:Nnodes
        g = x_surf(:,T_surf(nn,[2:end]));
        node_surf = x_surf(:,T_surf(nn,1)); 
        BasisVectors = g - repmat(node_surf,1,N_BasisVector);
        G = BasisVectors.' * BasisVectors;
        B = inv(G) * BasisVectors.';
        Bmatrix_surf(:,:,nn) = B;
        BasisVectors_surf(:,:,nn) = BasisVectors;    
        
        %p-space
        ind = T_pspace(nn,:);
        BasisVectors = par_points(ind(2:end),:) - repmat(par_points(ind(1),:),N_BasisVector,1); 
        BasisVectors = BasisVectors.';
        Bmatrix_pspace(:,:,nn) = inv(BasisVectors);
    end;


return;