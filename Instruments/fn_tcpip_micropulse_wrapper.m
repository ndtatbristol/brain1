function [info, h_fn_connect, h_fn_disconnect, h_fn_reset, h_fn_acquire, h_fn_send_options] = fn_tcpip_micropulse_wrapper(dummy)
info.name = 'Peak NDT Micropulse TCPIP';
echo_on = 0;

info.options_info.ip_address.label = 'IP address';
info.options_info.ip_address.default = '10.1.1.2';
info.options_info.ip_address.type = 'string';

info.options_info.port_no.label = 'Port number';
info.options_info.port_no.default = 1067;
info.options_info.port_no.type = 'int';
info.options_info.port_no.constraint = [1, 99999];
        
info.options_info.acquire_mode.label = 'Acquisition';
info.options_info.acquire_mode.default = 'HMC'; %Sample frequency (MHz); int; 1e6; {25, 50, 100}
info.options_info.acquire_mode.type = 'constrained';
info.options_info.acquire_mode.constraint = {'SAFT', 'FMC', 'HMC', 'CSM'};

info.options_info.sample_freq.label = 'Sample frequency (MHz)';
info.options_info.sample_freq.default = '25'; %Sample frequency (MHz); int; 1e6; {25, 50, 100}
info.options_info.sample_freq.type = 'constrained';
info.options_info.sample_freq.constraint = {'10', '25', '50', '100'};

info.options_info.pulse_voltage.label = 'Pulse voltage (V)';
info.options_info.pulse_voltage.default = 100;
info.options_info.pulse_voltage.type = 'double';
info.options_info.pulse_voltage.constraint = [50, 200];
info.options_info.pulse_voltage.multiplier = 1;

info.options_info.pulse_width.label = 'Pulse width (ns)';
info.options_info.pulse_width.default = 80e-9;
info.options_info.pulse_width.type = 'double';
info.options_info.pulse_width.constraint = [20e-9, 500e-9];
info.options_info.pulse_width.multiplier = 1e-9;

info.options_info.time_pts.label = 'Time points';
info.options_info.time_pts.default = 1000;
info.options_info.time_pts.type = 'int';
info.options_info.time_pts.constraint = [100, 3000];

info.options_info.sample_bits.label = 'Sample bits';
info.options_info.sample_bits.default = '8';
info.options_info.sample_bits.type = 'constrained';
info.options_info.sample_bits.constraint = {'8', '12', '16'};

info.options_info.db_gain.label = 'Gain (dB)';
info.options_info.db_gain.default = 40;
info.options_info.db_gain.type = 'int';
info.options_info.db_gain.constraint = [0, 70];%check!

info.options_info.filter_no.label = 'Filter number';
info.options_info.filter_no.default = 4;
info.options_info.filter_no.type = 'int';
info.options_info.filter_no.constraint = [0, 4];

info.options_info.prf.label = 'Maximum PRF (kHz)';
info.options_info.prf.default = 2e3;
info.options_info.prf.type = 'double';
info.options_info.prf.multiplier = 1e3;
info.options_info.prf.constraint = [0.001e3 20e3];

info.options_info.averages.label = 'Averages';
info.options_info.averages.default = 1;
info.options_info.averages.type = 'int';
info.options_info.averages.constraint = [1, 64];

info.options_info.gate_start.label = 'Time start (us)';
info.options_info.gate_start.default = 0;
info.options_info.gate_start.type = 'double';
info.options_info.gate_start.constraint = [0, 1e3];
info.options_info.gate_start.multiplier = 1e-6;

info.options_info.instrument_delay.label = 'Instrument delay (ns)';
info.options_info.instrument_delay.default = 0;
info.options_info.instrument_delay.type = 'double';
info.options_info.instrument_delay.constraint = [-1e6, 1e6];
info.options_info.instrument_delay.multiplier = 1e-9;

h_fn_acquire = @fn_acquire;
h_fn_send_options = @fn_send_options;
h_fn_reset = @fn_reset;
h_fn_disconnect = @fn_disconnect;
h_fn_connect = @fn_connect;
% h_fn_get_options = @fn_get_options;

options_sent = 0;
connected = 0;
% exp_data = [];
tx_no = [];
rx_no = [];
time_axis = [];
options_sent = 0;

    function exp_data = fn_acquire(dummy)
        exp_data = [];
        if ~options_sent
            %this should give a warning!
            return;
        end
        if ~connected
            return;
        end
        exp_data.time_data = fn_ag_do_test_tcpip;
        exp_data.tx = tx_no;
        exp_data.rx = rx_no;
        exp_data.time = time_axis;
    end

    function fn_send_options(options, no_channels)
        if ~connected
            return;
        end
        switch options.acquire_mode
            case 'SAFT'
                [options.tx_ch, options.rx_ch] = fn_set_fmc_input_matrices(no_channels, 0);
                options.rx_ch = options.tx_ch;
            case 'FMC'
                [options.tx_ch, options.rx_ch] = fn_set_fmc_input_matrices(no_channels, 0);
            case 'HMC'
                [options.tx_ch, options.rx_ch] = fn_set_fmc_input_matrices(no_channels, 1);
            case 'CSM'
                options.tx_ch = ones(1, no_channels);
                options.rx_ch = ones(1, no_channels);
        end
        options.sample_freq  = str2num(options.sample_freq) * 1e6;
        options.sample_bits = str2num(options.sample_bits);
        [tx_no, rx_no] = fn_ag_set_test_options_tcpip(options, echo_on);
        time_step = 1 / options.sample_freq;
        time_axis = [options.gate_start:time_step:options.gate_start + time_step*(options.time_pts-1)]' - options.instrument_delay;
        options_sent = 1;
    end

    function fn_reset(dummy)
        fn_ag_reset_tcpip(echo_on);
    end

    function res = fn_disconnect(dummy)
        fn_ag_disconnect_tcpip(echo_on);
        connected = 0;
        res = connected;
    end

    function res = fn_connect(options)
        connected = fn_ag_connect_tcpip(options.ip_address, options.port_no, echo_on);
        res = connected;
    end

end
