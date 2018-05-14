%Triangulation of rectangular domain in n-dimensional space 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Differencies from v0:
%1. returns list of nodes (nodes)
%2. returns list of elementary cubes (C)
%3. returns indices of triangles (index of cube which contains each triangle)

function [ nodes, T, C, T_ind_C ] = fn_ngrid_triangulation_v1( N )

if size(N,1)==1
    N = N';
end;

n = length(N);
[ ~, Tbin_cube ] = fn_ncube_triangulation(n);
Tcube = zeros(size(Tbin_cube));
for ii=1:size(Tbin_cube,1)
    for jj=1:size(Tbin_cube,2)
        pp = str2num(Tbin_cube{ii,jj}) + 1;
        Tcube(ii,jj) = sub2ind_my(N, pp);        
    end;
end;

Nadd = 1;
Np = size(Tbin_cube,1);
T = zeros(prod(N-1)*Np,size(Tbin_cube,2));
T(1:Np,:) = Tcube;
for ii=1:length(N) 
    for jj=2:N(ii)-1
        ind1 = [1:Np] + Np*(jj-1);        
        T(ind1,:) = T(ind1 - Np,:) + Nadd;
    end;
    Np = Np * (N(ii)-1);  
    Nadd = Nadd * N(ii);
end;

%define nodes
nodes = zeros(prod(N),length(N));
Np = 1;
nodes(1,:) = ones(1,length(N));
nodes(1:N(1),:) = [[1:N(1)]',ones(N(1),length(N)-1)];
for ii=1:length(N)
    Nadd = zeros(1,length(N)); Nadd(ii)=1;
    for jj=2:N(ii)
        ind1 = [1:Np] + Np*(jj-1);
        nodes(ind1,:) = nodes(ind1-Np,:) + repmat(Nadd,length(ind1-Np),1);
    end;
    Np = Np * N(ii);        
end;

%define cubes
Nadd = 1;
Np = 1;
C = zeros(prod(N-1),2^length(N));
C(1:Np,:) = sort(unique(Tcube(:)));
for ii=1:length(N) 
    for jj=2:N(ii)-1
        ind1 = [1:Np] + Np*(jj-1);        
        C(ind1,:) = C(ind1 - Np,:) + Nadd;
    end;
    Np = Np * (N(ii)-1);  
    Nadd = Nadd * N(ii);
end;

%define indices of triangles (index of cube which contains each triangle)
Nt_cube = size(Tcube,1);
T_ind_C = repmat([1:size(C,1)],Nt_cube,1);
T_ind_C = T_ind_C(:);

return;

%--------------------------------------------------------------------------

function linearInd = sub2ind_my(N, sub)

sub = sub(end:-1:1);

linearInd = sub(1);
for ii=length(N):-1:2
    linearInd = linearInd + (sub(ii)-1)*prod(N(1:ii-1));
end;


return;