function [tx_matrix, rx_matrix] = fn_set_fmc_input_matrices_advance(elements, fmc);

%no_elements is number of elements to use.
%fmc is 1 for full capture, 0 for half matrix capture

max_channels = 64;%max parallel channels on mp
if length(elements) > max_channels
    disp('Micropulse can only support 64 channels')
    return;
end
txs = zeros(max_channels,1);
txs(elements) = 1;
tx_matrix = diag(txs);
rx_matrix = zeros(size(tx_matrix));
rx_matrix(elements,elements) = 1;
time_traces = length(elements)^2;

if fmc == 0
    rx_matrix = triu(rx_matrix);
    time_traces = (length(elements)/2)*(length(elements)+1);
end
