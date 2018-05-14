function [tx_matrix, rx_matrix] = fn_set_instrument_tx_rx_matrices(no_elements, acquisition_mode)
%SUMMARY
%   Creates transmission and reception matrices to define instrument events
%   for various acquisition_mode
%AUTHOR
%   Paul Wilcox 26/4/2018
%INPUTS
%   no_elements - integer, number of elements in array
%   acquisition_mode - one of 'FMC', 'HMC', 'HMC1', 'HMC2', 'SAFT', 'CSM'
%OUTPUTS
%   tx_matrix - matrix where each row is a transmission event and each
%   column represents element in array, 1 = firing, 0 = off in that event
%   rx_matrix - matrix of same size specifying which elements are recording
%   for each transmission event

%NOTES
%   Intended to provide instrument-agnostic version of historic 
%   fn_set_fmc_input_matrices function



switch acquisition_mode
    case 'FMC'
        tx_matrix = diag(ones(no_elements,1));
        rx_matrix = ones(size(tx_matrix));
    case {'HMC', 'HMC1'}
        tx_matrix = diag(ones(no_elements,1));
        rx_matrix = triu(ones(size(tx_matrix)));
    case 'HMC2'
        tx_matrix = diag(ones(no_elements,1));
        rx_matrix = tril(ones(size(tx_matrix)));
    case 'SAFT'
        tx_matrix = diag(ones(no_elements,1));
        rx_matrix = diag(ones(no_elements,1));
    case 'CSM'
        tx_matrix = ones(1, no_elements);
        rx_matrix = ones(1, no_elements);
end

end
