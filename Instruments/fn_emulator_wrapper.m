function [info, h_fn_connect, h_fn_disconnect, h_fn_reset, h_fn_acquire, h_fn_send_options] = fn_emulator_wrapper(dummy)

file_exp_data = [];
exp_data = [];
mean_level = 0;
sample_bits = 16;
rel_noise_level = 1;
pause_length = 0;%0.5;
send_settings_pause = 0;%1;

if isdeployed
    fname_list = dir(fullfile(ctfroot, 'Instruments', 'Emulator data', '*.mat'));
else
    fname_list = dir(['Instruments', filesep, 'Emulator data', filesep, '*.mat']);
end
fname_list = {fname_list.name};

fname = fname_list{1};

info.name = 'Array controller emulator';
info.options_info.fname.label = 'Filename';
info.options_info.fname.default = fname;
info.options_info.fname.type = 'constrained';
info.options_info.fname.constraint = fname_list;

info.options_info.acquire_mode.label = 'Acquisition';
info.options_info.acquire_mode.default = 'HMC'; %Sample frequency (MHz); int; 1e6; {25, 50, 100}
info.options_info.acquire_mode.type = 'constrained';
info.options_info.acquire_mode.constraint = {'SAFT', 'FMC', 'HMC', 'CSM'};

info.options_info.rel_noise.label = 'Relative noise level';
info.options_info.rel_noise.default = 1;
info.options_info.rel_noise.multiplier = 1;
info.options_info.rel_noise.type = 'double';
info.options_info.rel_noise.constraint = [0, 1000];

info.options_info.time_pts.label = 'Time points';
info.options_info.time_pts.default = 4000;
info.options_info.time_pts.type = 'int';
info.options_info.time_pts.constraint = [100, 4000];

info.options_info.sample_bits.label = 'Sample bits';
info.options_info.sample_bits.default = '16';
info.options_info.sample_bits.type = 'constrained';
info.options_info.sample_bits.constraint = {'1', '2', '4', '8', '12', '16'};

info.options_info.gate_start.label = 'Time start (us)';
info.options_info.gate_start.default = 0;
info.options_info.gate_start.type = 'double';
info.options_info.gate_start.constraint = [0, 1e3];
info.options_info.gate_start.multiplier = 1;

h_fn_acquire = @fn_acquire;
h_fn_send_options = @fn_send_options;
h_fn_reset = @fn_reset;
h_fn_disconnect = @fn_disconnect;
h_fn_connect = @fn_connect;

options_sent = 0;
connected = 0;
tx_no = [];
rx_no = [];
time_axis = [];
options_sent = 0;

    function data = fn_acquire(dummy)
        data = exp_data;
        if ~connected
            return;
        end
        pause(pause_length);
        if ~options_sent
            %this should give a warning!
            disp('Options not sent');
            return;
        end
        data.time_data = data.time_data + randn(size(data.time_data)) * mean_level * rel_noise_level;
        if sample_bits > 1
            q = 2 ^ sample_bits - 1;
            q0 = 2 ^ (sample_bits - 1) - 1;
            data.time_data = floor((data.time_data + 1) / 2 * (1 - eps) * q) / q0 - 1;
        else
            data.time_data = round((data.time_data + 1) / 2) * 2 - 1;
        end
    end

    function fn_send_options(options, no_channels)
        if ~connected
            return;
        end
        pause(send_settings_pause);
        %load data file
        tmp = load(fullfile('Instruments', 'Emulator data', options.fname));
        
        file_exp_data = tmp.exp_data;
%         options.time_pts = length(file_exp_data.time);
        
        %generate FMC data set - NB, data in these files must be either FMC
        %or HMC (not CSM or SAFT)
        if length(file_exp_data.tx < length(unique(file_exp_data.tx)) ^ 2)
            file_exp_data = fn_expand_hmc_to_fmc(file_exp_data);
        end
        mean_level = mean(mean(abs(file_exp_data.time_data)));
        sample_bits = str2num(options.sample_bits);
        exp_data = file_exp_data;

        if isempty(file_exp_data)
            return;
        end
        if ~isfield(file_exp_data, 'array')
            return;
        end
        switch options.acquire_mode
            case 'SAFT'
                jj = find(file_exp_data.tx == file_exp_data.rx);
                exp_data.time_data = exp_data.time_data(:, jj);
                exp_data.tx = file_exp_data.tx(jj);
                exp_data.rx = file_exp_data.rx(jj);
            case 'FMC'
                %do nothing - data is already FMC!
            case 'HMC'
                tmp = fn_transducer_pairs(length(unique(file_exp_data.tx)), 1);
                exp_data.time_data = zeros(size(file_exp_data.time_data, 1), size(tmp, 1));
                exp_data.tx = zeros(1,size(tmp, 1));
                exp_data.rx = zeros(1,size(tmp, 1));
                for ii = 1:size(tmp, 1)
                    jj = find(file_exp_data.tx == tmp(ii, 1) & file_exp_data.rx == tmp(ii, 2));
                    exp_data.time_data(:, ii) = file_exp_data.time_data(:, jj);
                    exp_data.tx(ii) = tmp(ii, 1);
                    exp_data.rx(ii) = tmp(ii, 2);
                end
            case 'CSM'
                unique_rx = unique(exp_data.rx);
                exp_data.tx = ones(1, length(unique_rx));
                exp_data.rx = unique_rx;
                exp_data.time_data = zeros(length(exp_data.time), length(unique_rx));
                for ii = 1:length(unique_rx)
                    exp_data.time_data(:,ii) = sum(file_exp_data.time_data(:, find(file_exp_data.rx == unique_rx(ii))), 2);
                end;
        end
        rel_noise_level = options.rel_noise;
        if options.time_pts < length(exp_data.time)
            exp_data.time_data = exp_data.time_data(1:options.time_pts, :);
            exp_data.time = exp_data.time(1:options.time_pts);
        end
        options_sent = 1;
    end

    function fn_reset(dummy)
    end


    function res = fn_disconnect(dummy)
        pause(pause_length);
        connected = 0;
        res = connected;
    end

    function res = fn_connect(dummy)
        pause(pause_length);
        connected = 1;
        res = connected;
    end

end


function fmc_exp_data = fn_expand_hmc_to_fmc(hmc_exp_data)
fmc_exp_data = hmc_exp_data;
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
end