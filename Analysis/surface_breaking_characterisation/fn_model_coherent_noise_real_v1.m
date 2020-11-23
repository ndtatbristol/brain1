%rand noise and smoothing is applied to the full S-matrix first. 
%Then S-matrix is converted into the exp format
function [S_noise,S_noise_pc] = fn_model_coherent_noise_real_v1(sigma,corr_length,N_test,phi_in,phi_sc,phi_exp,V,ind_S_phi)

NN = length(phi_in)*length(phi_sc);
[p1,p2] = meshgrid(phi_in,phi_sc);
% ind_p1p2 = find(p2>=p1);
for i1=1:length(phi_in)
    f1{i1} = exp(-(p1-phi_in(i1)).^2/corr_length(1)^2);
end;
for i1=1:length(phi_sc)
    f2{i1} = exp(-(p2-phi_sc(i1)).^2/corr_length(2)^2);
end;

fg = zeros(NN,NN);
nn=0;
for i1=1:length(phi_in)
    for i2=1:length(phi_sc)
        nn = nn+1;
        tmp = f1{i1}.*f2{i2};
        fg(nn,:) = tmp(:);
    end;
end;

ng_real = normrnd(0,1,NN,N_test);
ngl_real = fg*ng_real;
ngl_real = ngl_real-repmat(mean(ngl_real),NN,1); ngl_real = ngl_real./repmat(std(ngl_real),NN,1)*sigma; 

S_noise = ngl_real;

%convert into exp format
[a1,a2] = meshgrid(phi_exp,phi_exp);
NN_exp = length(ind_S_phi);
S_noise_exp = zeros(NN_exp,N_test);
h = waitbar(0,'Calculating noise...');
for ii=1:N_test
    S_tmp = reshape(S_noise(:,ii),length(phi_sc),length(phi_in));
    S_tmp_exp = interp2(p1,p2,S_tmp,a1,a2,'spline');
    S_noise_exp(:,ii) = S_tmp_exp(ind_S_phi);
    waitbar(ii/N_test);
end;
close(h);

%PC-space
S_noise_pc = V.'*S_noise_exp;

return;
