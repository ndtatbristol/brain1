function [tx_matrix, rx_matrix] = fn_set_fmc_input_matrices(no_elements, hmc);

%no_elements is number of elements to use.
%fmc is 1 for full capture, 0 for half matrix capture

max_channels = 128;%max parallel channels on mp
if no_elements > max_channels
    disp('Micropulse can only support 128 channels')
    return;
end

tx_matrix = diag(ones(no_elements,1));
rx_matrix = zeros(size(tx_matrix));
rx_matrix(1:end,1:no_elements) = 1;
time_traces = no_elements^2;

if hmc == 1
    rx_matrix = triu(rx_matrix);
    time_traces = (no_elements/2)*(no_elements+1);
end
if hmc == 2
    rx_matrix = tril(rx_matrix);
    time_traces = (no_elements/2)*(no_elements+1);
end