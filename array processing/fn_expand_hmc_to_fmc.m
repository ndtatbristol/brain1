function fmc_exp_data = fn_expand_hmc_to_fmc(hmc_exp_data)
%SUMMARY
%   Expands HMC array data to FMC data

%--------------------------------------------------------------------------
fmc_exp_data = hmc_exp_data;
%check it is HMC data in the first place
switch fn_determine_exp_data_type(hmc_exp_data)
    case 'HMC'
        %do nothing
    case 'FMC'
        %do nothing and exit function- it already is FMC
        return
    otherwise
        error('Half Matrix Capture (HMC) data expected');
end

unique_rx = unique(hmc_exp_data.rx);

fmc_exp_data.time_data = zeros(length(hmc_exp_data.time), length(unique_rx) ^ 2);
fmc_exp_data.tx = zeros(1, length(unique_rx) ^ 2);
fmc_exp_data.rx = zeros(1, length(unique_rx) ^ 2);
kk = 1;
for ii = 1:length(unique_rx)
    for jj = 1:length(unique_rx)
        mm = find(hmc_exp_data.tx == unique_rx(ii) & hmc_exp_data.rx == unique_rx(jj));
        if isempty(mm)
            mm = find(hmc_exp_data.rx == unique_rx(ii) & hmc_exp_data.tx == unique_rx(jj));
        end
        if isempty(mm)
            errordlg('Invalid data file. Data for emulator must be HMC or FMC.');
            fmc_exp_data = [];
            return;
        end
        fmc_exp_data.time_data(:, kk) = hmc_exp_data.time_data(:, mm);
        fmc_exp_data.tx(kk) = unique_rx(ii);
        fmc_exp_data.rx(kk) = unique_rx(jj);
        kk = kk + 1;
    end
end