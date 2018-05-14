 function [ T, Tbin ] = fn_ncube_triangulation( n )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

T1 = 0; T2 = 1;
for in=2:n
    T11 = [ [zeros(size(T1,1),1),T1]; [ones(size(T2,1),1),T2] ];    
    T12 = [ [zeros(size(T2,1),1),T2]; [ones(size(T1,1),1),T1] ];   
    T1 = T11; T2 = T12;
end;

T = zeros(size(T1,1),size(T1,2)+1);

for ii=1:size(T1,1)
    tmp1 = T1(ii,:); 
    Tbin{ii,1} = num2str(T1(ii,:));
    T(ii,1) = bin2dec(num2str(T1(ii,:)))+1;
    for jj=1:size(T1,2)
        jj1 = size(T1,2)-jj+1; %so jj1-th basis vector corresponds to the jj1-th parameter
        tmp2 = tmp1;
        tmp2(jj) = ~tmp2(jj);
        Tbin{ii,jj1+1} = num2str(tmp2);
        T(ii,jj1+1) = bin2dec(num2str(tmp2))+1;
    end;    
end;

return;

