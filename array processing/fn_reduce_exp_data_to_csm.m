function csm_exp_data = fn_reduce_exp_data_to_csm(orig_exp_data)
%SUMMARY
%   Reduces FMC or HMC array data to CSM data

%if HMC data, expand to FMC first
%--------------------------------------------------------------------------
csm_exp_data = orig_exp_data;
csm_exp_data = fn_expand_hmc_to_fmc(csm_exp_data);

unique_rx = unique(csm_exp_data.rx);
csm_exp_data.tx = ones(1, length(unique_rx));
csm_exp_data.rx = unique_rx;
csm_exp_data.time_data = zeros(length(csm_exp_data.time), length(unique_rx));
for ii = 1:length(unique_rx)
    csm_exp_data.time_data(:,ii) = sum(orig_exp_data.time_data(:, find(orig_exp_data.rx == unique_rx(ii))), 2);
end;
end