function [info, h_fn_connect, h_fn_disconnect, h_fn_reset, h_fn_acquire, h_fn_send_options] = fn_verasonics_wrapper(dummy)
global Trans
global Resource
global TW
global TX
global Event
global Receive
global SeqControl
global TGC
global VDASupdates
global VDAS
global Control
global VSX_Control
global array
global ph_velocity
global RcvProfile
global TPC


info.name = 'Verasonics';
echo_on = 0;

info.options_info.acquire_mode.label = 'Acquisition';
info.options_info.acquire_mode.default = 'FMC'; %Sample frequency (MHz); int; 1e6; {25, 50, 100}
info.options_info.acquire_mode.type = 'constrained';
info.options_info.acquire_mode.constraint = {'FMC','CSM','HADAMARD'};

info.options_info.sample_freq.label = 'Sample frequency (MHz)';
info.options_info.sample_freq.default = '25'; %Sample frequency (MHz); int; 1e6; {25, 50, 100}
info.options_info.sample_freq.type = 'constrained';
info.options_info.sample_freq.constraint = {'10.8696', '25', '50', '62.5', '125','250'};

info.options_info.pulse_voltage.label = 'Pulse voltage (V)';
info.options_info.pulse_voltage.default = 1.6;
info.options_info.pulse_voltage.type = 'double';
info.options_info.pulse_voltage.constraint = [1, 96];
info.options_info.pulse_voltage.multiplier = 1;

info.options_info.pulse_width.label = 'Pulse length (cycles)';
info.options_info.pulse_width.default = 1;
info.options_info.pulse_width.type = 'double';
info.options_info.pulse_width.constraint = [1, 10];
info.options_info.pulse_width.multiplier = 1;

info.options_info.pulse_freq.label = 'Pulse frequency (MHz)';
info.options_info.pulse_freq.default = 5;
info.options_info.pulse_freq.type = 'double';
info.options_info.pulse_freq.constraint = [1, 15];
info.options_info.pulse_freq.multiplier = 1;

info.options_info.time_pts.label = 'Time points';
info.options_info.time_pts.default = 1000;
info.options_info.time_pts.type = 'int';
info.options_info.time_pts.constraint = [100, 20000];

info.options_info.db_gain.label = 'Gain (dB)';
info.options_info.db_gain.default = 25;
info.options_info.db_gain.type = 'int';
info.options_info.db_gain.constraint = [-1, 54];%set to 

info.options_info.prf.label = 'Maximum PRF (kHz)';
info.options_info.prf.default = 2e3;
info.options_info.prf.type = 'double';
info.options_info.prf.multiplier = 1e3;
info.options_info.prf.constraint = [0.001e3 20e3];

info.options_info.gate_start.label = 'Time start (us)';
info.options_info.gate_start.default = 0;
info.options_info.gate_start.type = 'double';
info.options_info.gate_start.constraint = [-50, 1e3];
info.options_info.gate_start.multiplier = 1e-6;

info.options_info.averages.label = 'Averages';
info.options_info.averages.default = 1;
info.options_info.averages.type = 'int';
info.options_info.averages.constraint = [1, 64];

info.options_info.attenuation.label = 'Attenuation (dB/us)';
info.options_info.attenuation.default = 0;
info.options_info.attenuation.type = 'double';
info.options_info.attenuation.constraint = [0, 100];
info.options_info.attenuation.multiplier = 1;

info.options_info.attenuation_end.label = 'Attenuation end (us)';
info.options_info.attenuation_end.default = 0;
info.options_info.attenuation_end.type = 'double';
info.options_info.attenuation_end.constraint = [0, 1e3];
info.options_info.attenuation_end.multiplier = 1;

info.options_info.instrument_delay.label = 'Instrument delay (ns)';
info.options_info.instrument_delay.default = 0;
info.options_info.instrument_delay.type = 'double';
info.options_info.instrument_delay.constraint = [-1e6, 1e6];
info.options_info.instrument_delay.multiplier = 1;

h_fn_acquire = @fn_acquire;
h_fn_send_options = @fn_send_options;
h_fn_reset = @fn_reset;
h_fn_disconnect = @fn_disconnect;
h_fn_connect = @fn_connect;
% h_fn_get_options = @fn_get_options;

options_sent = 0;
connected = 0;
options_sent = 0;
exp_data = [];
hadamard_excite = 0;
big_inv_s = [];

    function exp_data = fn_acquire(options)
        exp_data = [];
        if ~options_sent
            %this should give a warning!
            return;
        end
        if ~connected
            return;
        end
        evalin('base','runAcq(VSX_Control);')
        evalin('base','runAcq(Control);')
        
        capData=evalin('base','RcvData;');
        [exp_data]=fn_verasonics_convert_short(Trans, Receive, capData);
        if strcmp(options.acquire_mode,'CSM')
           exp_data.tx= ones(1, length(array.el_xc));
           exp_data.rx= [1: length(array.el_xc)];
        end
        exp_data.time_data=exp_data.time_data./options.averages;
        if hadamard_excite
            exp_data.raw_data = exp_data.time_data;
            exp_data.time_data = exp_data.time_data * big_inv_s;
        end
        exp_data.array=array;
        exp_data.time=exp_data.time-options.instrument_delay.*1e-9;
        if options.gate_start<0
            exp_data.time=exp_data.time+options.gate_start;
        end
    end

    function fn_send_options(options, no_channels)
        TX=[];
        Event=[];
        Receive=[];
        TPC=[];
        TW=[];
        if ~connected
            return;
        end
        switch options.acquire_mode
            case 'FMC'
                hadamard_excite = 0;
                [options.tx_ch, options.rx_ch] = fn_set_fmc_input_matrices(no_channels, 0);
            case 'CSM'
                hadamard_excite = 0;
                options.tx_ch = ones(1, no_channels);
                options.rx_ch = ones(1, no_channels);
            case 'HADAMARD'
                hadamard_excite = 1;
                [options.tx_ch, options.rx_ch] = fn_set_fmc_input_matrices(no_channels, 0);
                h = hadamard(2^nextpow2(no_channels+1));
                s = h(2:end, 2:end);
                s = s(1:no_channels, 1:no_channels);
                options.tx_ch = s;
                s_inv = inv(s);
                n = size(s_inv, 2);
                r = repmat([1: n ^ 2]',n, 1);
                c = repmat([1: n]',n, n) + repmat([0: n - 1] * n, n ^ 2, 1);
                big_inv_s = repmat(s_inv(:)',[n,1]);
                big_inv_s = big_inv_s(:);
                big_inv_s = sparse(r(:), c(:), big_inv_s(:), n ^ 2, n ^ 2);
        end
        options.sample_freq  = str2num(options.sample_freq) * 1e6;
        options.pulse_freq = options.pulse_freq .*1e6;
        if options.gate_start<0
            options.gate_start=round(options.gate_start/4e-9).*4e-9;
        end
        
        options.attenuation_end=options.attenuation_end.*1e-6;%if go to microseconds need 1e-6 factor
        options.attenuation=options.attenuation.*1e-6;
        %keyboard
        evalin('base','RcvData=[];');
        [RcvProfile, Control, VSX_Control] = fn_set_test_options_verasonics(options, ph_velocity, array);
        
        evalin('base','runAcq(VSX_Control);')
        [result, Trans.use_volts] = setTpcProfileHighVoltage(options.pulse_voltage,1);
        runAcq(Control(3))
        Control=Control(1:2);
          
        options_sent = 1;
    end

    function fn_reset(dummy)
       
    end

    function res = fn_disconnect(dummy)
        Result = hardwareClose();
        connected = 0;
        res = connected;
    end

    function res = fn_connect(options)
        %keyboard
        if ~exist('array','var')
            warndlg('Select array first','Warning')
            connected = 0;
        elseif ~exist('ph_velocity','var')
            warndlg('Select material first','Warning')
            connected = 0;
        else
            connected = 1;
        end
        res = connected;
    end

end
