function [x,y,Nsamp,success] = fn_Poisson_disk_sampling(N,rho,r_Nyq,N_term)
%USAGE
%   [x,y,Nsamp,success] = fn_Poisson_disk_sampling(N,rho,r_Nyq,N_term);
%AUTHOR
%	Alexander Velichko (2010)
%SUMMARY
%   Generates values for transducer positions under Poisson disk
%   constraints
%INPUTS
%   N - number of points
%   r_Nyq - minimum inter-element spacing
%   0<rho<=1 - parameter, actual element spacing is rho*r_Nyq e.g. 0.8
%   N_term  - maximum number of attempts for each point
%OUTPUTS
%   x - transducer x position (centred on origin)
%   y - transducer y position (centred on origin)
%   Nsamp - number of attempts before introduction of a new element
%   success - if no success try either lower rho or higher N_term
%NOTES
%  

success = 1;
k_Nyq = 1/r_Nyq;
W = pi*k_Nyq;
r_hex = 2/sqrt(3)*r_Nyq;
r_max = rho*r_hex;
R = 1/W * sqrt(N*pi*2/sqrt(3)); %minimum radius for closest packed distribution of N points with r_Nyq minimum spacing
x = []; y = [];
Nsamp(1) = 1;
for ii=1:N
%     disp(sprintf('%i',ii));
    fl = 0;
    for jj=1:N_term
        p = rand(2,1);
        p = (p - 0.5)*2*R;
        rp = sqrt(p(1)^2 + p(2)^2);
        if rp <= R
            if ii==1
               x = p(1); y = p(2); fl = 1;
            else
               rpp = sqrt((p(1) - x).^2 + (p(2) - y).^2);
               if rpp >= r_max
                   x = [x;p(1)]; y = [y;p(2)];
                   Nsamp(ii) = jj;
                   fl = 1;
               end;
            end;
        end;            
        if fl 
            break;
        end;
    end;  
    if ~fl
        disp(sprintf('Sampling process terminated, N=%.3f', ii)); 
        success = 0;
        return;
    end;
end;

return;