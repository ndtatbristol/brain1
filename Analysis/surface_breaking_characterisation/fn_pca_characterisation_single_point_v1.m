%new version 16-05-2016
%when there is not normal to the surface - find minimum distance; recursive implementation
function [par_test, PS_test_nodes, PS_test_project_dist, PS_test_project]  = fn_pca_characterisation_single_point_v1(PS_test, PS_database, ...
    T_database, PC_BasisVectors_database, PC_Bmatrix_database, par_grid_database )


%for a given testing point find a point on the surface that the distance is
%the shortest; 
[ind_surf_el, x_project_surf_el, x_dist_surf] = fn_projection_onto_surface(PS_test, PS_database, ...
    T_database, PC_BasisVectors_database, PC_Bmatrix_database);

PS_test_nodes = ind_surf_el;
PS_test_project_dist = x_dist_surf;

%projection point
BasisVectors = squeeze(PC_BasisVectors_database(:,:,ind_surf_el));
PS_test_project = PS_database(:,T_database(ind_surf_el,1)) + BasisVectors * x_project_surf_el; 

%interpolation
par_grid_database_surf_el = par_grid_database(T_database(ind_surf_el,:),:);
par_test = par_grid_database_surf_el(1,:).' + ...
          ( par_grid_database_surf_el(2:end,:).' - repmat(par_grid_database_surf_el(1,:).',1,size(T_database,2)-1) ) * x_project_surf_el;

return;
