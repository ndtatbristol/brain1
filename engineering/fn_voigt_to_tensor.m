function C = fn_voigt_to_tensor(D)
%SUMMARY
%   Converts 6x6 stiffness matrix into 3x3x3x3 stiffness tensor
%USAGE
%   C = fn_voigt_to_tensor(D)
%INPUTS
%   D - 6x6 stiffness matrix
%OUTPUTS
%   C - 3x3x3x3 stiffness tensor
%AUTHOR
%   Paul Wilcox (2010)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = zeros(3,3,3,3);
for ii = 1:3
    for jj = 1:3
        for kk=1:3
            for ll = 1:3
                C(ii,jj,kk,ll) = D(fn_ij_to_reduced(ii, jj), fn_ij_to_reduced(kk, ll));
            end;
        end;
    end;
end;
return;

function rr = fn_ij_to_reduced(ii, jj)
if ii == jj
    rr = ii;
    return;
else
    if ii~=1 & jj~=1
        rr = 4;
        return;
    elseif ii~=2 & jj~=2
        rr = 5;
        return;
    elseif ii~=3 & jj~=3
        rr = 6;
    end;
end;
return;