function [sigma]=fn_rayleigh_mle(x)
    % Maximum liklihood estimator (biased) of Rayleigh parameter (equivalent to raylfit() in statistics toolbox)
    sigma = sqrt(sum(x.^2)/(2*length(x)));
    
end