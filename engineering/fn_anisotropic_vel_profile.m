function [vph, vgr, p] = fn_anisotropic_vel_profile(stiffness_matrix_or_tensor, rho, n)
%SUMMARY
%   Computes phase and group velocity vectors in anisotropic material 
%   (specified by stiffness matrix or tensor, stiffness_matrix_or_tensor, 
%   and density, rho) for phase velocity direction specified by n
%INPUTS
%   stiffness_matrix_or_tensor - 6x6 stiffness matrix or 3x3x3x3 tensor
%   rho - density (scalar)
%   n - Nx3 matrix of unit vector directions that specify phase
%   velocity direction
%OUTPUTS
%   vph - Nx3xM matrix of phase velocity vectors (each row will point in same
%   direction as equivalent row of n) for each mode M
%   vgr - Nx3xM matrix of corresponding group velocity vectors
%   p - Mx3 matrix of polarisation vectors for each mode M
%NOTES
%   vgr vectors are group velocity vectors for corresponding phase velocity
%   directions, the latter being in the direction of n. Hence, vgr is NOT 
%   the group velocity in the direction of n.
%--------------------------------------------------------------------------

%Input checks
if size(n, 2) ~= 3
    error('n must be Nx3 matrix')
end
if (ndims(stiffness_matrix_or_tensor) == 2 && ~all(size(stiffness_matrix_or_tensor) == 6)) || ...
        (ndims(stiffness_matrix_or_tensor) == 4 && ~all(size(stiffness_matrix_or_tensor) == 3))
    error('stiffness_matrix_or_tensor must be 6x6 matrix or 3x3x3x3 tensor')
end
if ~isscalar(rho)
    error('rho must be a scalar')
end

%Force n to be unit vectors
n = n ./ repmat(sqrt(sum(n .^ 2, 2)), [1, 3]);

%Convert C to tensor
if ndims(stiffness_matrix_or_tensor) == 2
    C = fn_voigt_to_tensor(stiffness_matrix_or_tensor);
else
    C = stiffness_matrix_or_tensor;
end

N = size(n, 1);

%Prepare outputs
vph = zeros(N, 3, 3);
vgr = zeros(N, 3, 3);
p = zeros(N, 3, 3);

for nn = 1:N %loop over phase velocity directions
    %Prepare Christoffel equation matrix
    Cnn = zeros(3);
    for i = 1:3
        for j = 1:3
            for k = 1:3
                for l = 1:3
                    Cnn(i,k) = Cnn(i,k) + C(i,j,k,l) * n(nn,j) * n(nn,l);
                end
            end
        end
    end
    %Solve Eigenvalue problem to get phase velocity magnitude
    [V, D] = eig(Cnn);
    D = diag(D);
    
    %sort in ascending order to keep mode order consistent
    [~, i] = sort(D);
    D = D(i);
    V = V(:, i);
    
    c = sqrt(D / rho);
    for mm = 1:3
        %Split phase velocity into vector components
        vph(nn, :, mm) = c(mm) * n(nn, :);
        %Create polarisation vector from Eigenvectors
        p(nn, :, mm) = V(:, mm);
        %Calculate group velocity vector components
        for i = 1:3
            for j = 1:3
                for k = 1:3
                    for l = 1:3
                        vgr(nn, j, mm) = vgr(nn, j, mm) + C(i,j,k,l) * n(nn, k) * p(nn, i, mm) * p(nn, l, mm) / (rho * c(mm));
                    end
                end
            end
        end
    end
end

end


