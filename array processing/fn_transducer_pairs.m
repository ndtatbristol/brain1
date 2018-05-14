function trans_pairs=fn_transducer_pairs(no_of_trans, varargin);
%USAGE
%   trans_pairs = fn_transducer_pairs(no_of_trans [,include_pulse_echo]);
%AUTHOR
%	Anthony Croxford (2010), pulse_echo switch added by PDW 2010, FMC
%	switch in 2013
%SUMMARY
%   Generates values for the various transducer pairs (non-pulse echo
%   unless second argument is non-zero).
%INPUTS
%   no_of_trans - number of transducers
%OUTPUTS
%   trans_pairs - 2 column matrix where rows represent every transmitter-
%   receiver combination in half-matrix capture with or without pulse-echo
%   data according to value of include_pulse_echo

if nargin < 2
    include_pulse_echo = 0;
else
    include_pulse_echo = varargin{1};
end

if nargin < 3
    fmc = 0;
else
    fmc = varargin{2};
end

if fmc
    trans_grid = ones(no_of_trans);
    if ~include_pulse_echo
        trans_grid = trans_grid - eye(no_of_trans);
    end
else
    if include_pulse_echo
        trans_grid = triu(ones(no_of_trans));
    else
        trans_grid = triu(ones(no_of_trans), 1);
    end
end
[row, col]=find(trans_grid);
trans_pairs=[row, col];
trans_pairs=sortrows(trans_pairs);

end