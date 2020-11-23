function [S_crack_database, par_database] = fn_prepare_global_database_surface_crack_single_freq(path,lim_crack_length,lim_crack_angle)

%load data
load(path);

crack_angle_max = max(lim_crack_angle);
crack_angle_min = min(lim_crack_angle);

crack_length = S_surface_crack.crack_length;
ind_crack_length = find( crack_length>=min(lim_crack_length) & crack_length<=max(lim_crack_length));
crack_length = crack_length(ind_crack_length);
Nl = length(crack_length);

crack_angle = S_surface_crack.crack_angle;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%database with positive crack angles
if crack_angle_max>=0
    ind_ang_pos = find(crack_angle <= crack_angle_max & crack_angle >= crack_angle_min);
    [crack_angle_pos,ind_sort] = sort(crack_angle(ind_ang_pos));
    ind_ang_pos = ind_ang_pos(ind_sort);
    Na_pos = length(crack_angle_pos);
    nn = 0;
    for ii=1:Na_pos
        for jj=1:Nl
            nn = nn+1;
            S_pos{nn} = S_surface_crack.SL{ind_crack_length(jj),ind_ang_pos(ii)};             
        end;
    end;
else
    S_pos = [];
    crack_angle_pos = [];
end;

if crack_angle_min<0
    ind_ang_neg = find(-crack_angle <= crack_angle_max & -crack_angle>=crack_angle_min & -crack_angle<0);
    [crack_angle_neg,ind_sort] = sort(-crack_angle(ind_ang_neg));
    ind_ang_neg = ind_ang_neg(ind_sort); 
    Na_neg = length(crack_angle_neg);
    nn = 0;
    for ii=1:Na_neg
        for jj=1:Nl
            nn = nn+1;
            S_tmp = S_surface_crack.SL{ind_crack_length(jj),ind_ang_neg(ii)};
            S_tmp = S_tmp(end:-1:1,:); S_tmp = S_tmp(:,end:-1:1);
            S_neg{nn} = S_tmp;            
        end;
    end;
else
    S_neg = [];
    crack_angle_neg = [];
end;

S_crack_database.S = [S_neg, S_pos];
S_crack_database.phi_in = S_surface_crack.phi_in;
S_crack_database.phi_sc = S_surface_crack.phi_sc;

crack_angle = [crack_angle_neg; crack_angle_pos];
Na = length(crack_angle);

S_global_max = zeros(length(S_crack_database.S),1);
for ii=1:length(S_crack_database.S)
%    S_crack_database.freq{ii} = 1;
   S_global_max(ii) = max(max(abs(S_crack_database.S{ii}))); 
end;
S_crack_database.S_global_max = S_global_max;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%parameter space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par_dim_surf = [Nl, Na];
par_surf{1} = crack_length;
par_surf{2} = crack_angle;
[p1,p2] = ndgrid([1:Nl],[1:Na]);
par_ind_surf = [p1(:), p2(:)];


%normalised p-space
par_min = zeros(length(par_dim_surf),1);
par_max = zeros(length(par_dim_surf),1);
for ii=1:length(par_surf)
    par_min(ii) = min(par_surf{ii});
    par_max(ii) = max(par_surf{ii}); 
    par_surf_norm{ii} = (par_surf{ii}-par_min(ii)) / (par_max(ii)-par_min(ii));
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

par_database.par_surf = par_surf; 
par_database.par_surf_norm = par_surf_norm;
par_database.par_ind_surf = par_ind_surf; 

return;