function [ PS_test ] = fn_convert_Smatrix_into_PC_space( S_test, norm_coef_S_database, mean_S_database, V )

%convert S_test into PC space
%normalise
S_test = S_test / norm_coef_S_database;
%subscribe mean
S_test = S_test - repmat(mean_S_database,1,size(S_test,2));
PS_test = V' * S_test;


return;

