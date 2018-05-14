function [info, h_fn_connect, h_fn_disconnect, h_fn_reset, h_fn_acquire, h_fn_send_options] = fn_DSL_wrapper(dummy)
info.name = 'Diagnostic Sonar FI Toolbox';
echo_on = 0;
        
info.options_info.acquire_mode.label = 'Acquisition';
info.options_info.acquire_mode.default = 'HMC'; 
info.options_info.acquire_mode.type = 'constrained';
info.options_info.acquire_mode.constraint = {'FMC','HMC'};

info.options_info.sample_freq.label = 'Sample frequency (MHz)';
info.options_info.sample_freq.default = '40'; %Sample frequency (MHz); int; 1e6; {25, 50, 100}
info.options_info.sample_freq.type = 'constrained';
info.options_info.sample_freq.constraint = {'20','40'};

info.options_info.pulse_voltage.label = 'Pulse voltage (V)';
info.options_info.pulse_voltage.default = 80;
info.options_info.pulse_voltage.type = 'double';
info.options_info.pulse_voltage.constraint = [0.1, 150];%check
info.options_info.pulse_voltage.multiplier = 1;

info.options_info.pulse_frequency.label = 'Excitation Pulse Frequency (MHz)';
info.options_info.pulse_frequency.default = 5;
info.options_info.pulse_frequency.type = 'double';
info.options_info.pulse_frequency.constraint = [0.1, 20];%check
info.options_info.pulse_frequency.multiplier = 1;

info.options_info.pulse_cycles.label = 'Number of Excitation Pulse Cycles';
info.options_info.pulse_cycles.default = 0.5;
info.options_info.pulse_cycles.type = 'double';
info.options_info.pulse_cycles.constraint = [0.5, 10];%check
info.options_info.pulse_cycles.multiplier = 1;

info.options_info.pulse_active.label = '% of Excitation Pulse Actice';
info.options_info.pulse_active.default = 100;
info.options_info.pulse_active.type = 'double';
info.options_info.pulse_active.constraint = [0.1, 100];%check
info.options_info.pulse_active.multiplier = 1;

% info.options_info.pulse_width.label = 'Pulse width (ns)';
% info.options_info.pulse_width.default = 80e-9;
% info.options_info.pulse_width.type = 'double';
% info.options_info.pulse_width.constraint = [1e-9, 100e-9];%check!
% info.options_info.pulse_width.multiplier = 1e-9;

info.options_info.time_pts.label = 'Time points';
info.options_info.time_pts.default = 1000;
info.options_info.time_pts.type = 'int';
info.options_info.time_pts.constraint = [1, 4000];%check

info.options_info.db_gain.label = 'Gain (dB)';
info.options_info.db_gain.default = 28;
info.options_info.db_gain.type = 'int';
info.options_info.db_gain.constraint = [0, 93];%check!

info.options_info.prf.label = 'Maximum PRF (kHz)';
info.options_info.prf.default = 2e3;
info.options_info.prf.type = 'double';
info.options_info.prf.multiplier = 1e3;
info.options_info.prf.constraint = [0.1e3 10e3];%check!

info.options_info.gate_start.label = 'Time start (us)';
info.options_info.gate_start.default = 2e-6;
info.options_info.gate_start.type = 'double';
info.options_info.gate_start.constraint = [0, 1e3];%check
info.options_info.gate_start.multiplier = 1e-6;

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

    function exp_data = fn_acquire(dummy)
        exp_data = [];
        if ~options_sent
            %this should give a warning!
            return;
        end
        if ~connected
            return;
        end
        
        if (sqrt(length(tx_no))-floor(sqrt(length(tx_no)))) ~= 0
            is_hmc = 1; 
        else
            is_hmc = 0;
        end
        
        i16time_data = fn_DSL_do_test((length(time_axis)*length(tx_no)),length(tx_no),length(time_axis),tx_no, rx_no, is_hmc, echo_on);
        exp_data.time_data = double(i16time_data);
        exp_data.time = time_axis;
        exp_data.tx = tx_no;
        exp_data.rx = rx_no;
    end

    function fn_send_options(options, no_channels)
        if ~connected
            return;
        end
        %Array details - all NDT lab arrays should have a corresponding file in the
        %NDT library!
        array_fname = 'Imasonic 1D 64els 2.00MHz 1.57mm pitch.mat'; %Needed to setup some things in DSL CFG file but no longer effects acquisition

        %Ultrasonic data (required for imaging and will be stored in exp_data file)
        ph_velocity = 6300; %Needed to setup some things in DSL CFG file but no longer effects acquisition
        
        %Setup other parameters required by processing code
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
        [tx_no, rx_no] = fn_DSL_define_fmc(options.tx_ch, options.rx_ch, echo_on);
        time_step = 1 / (str2double(options.sample_freq)*1e6);
        time_axis = [options.gate_start:time_step:options.gate_start + time_step*(options.time_pts-1)]';

        %Save data information if required
%         
%         setupfilename ='C:\DSL FIT\Setups\DSL Test Setup.cfg';
%         Csetupfilename ='C:\\DSL FIT\\Setups\\DSL Test Setup.cfg';
        setupfilename ='C:\Users\Public\Documents\FIToolbox\Configs\System\BRAIN Setup.cfg';
        Csetupfilename ='C:\\Users\\Public\\Documents\\FIToolbox\\Configs\\System\\BRAIN Setup.cfg';
        
        %Load the array file and set field in exp_data
        tmp = load(array_fname);
        exp_data.array = tmp.array;

        %Set detailed test options for DSL hardware
        %create setup file
        fn_DSL_create_cfg(setupfilename, exp_data, ph_velocity, options, echo_on);
        %load onto system
        fn_DSL_load_setup(Csetupfilename, echo_on);
        
        options_sent = 1;
    end

    function fn_reset(dummy)
%         fn_ag_reset(echo_on);
    end

    function res = fn_connect(options)
        fig = gcf;
        connected = fn_DSL_connect(echo_on);
        res = connected;
        commandwindow;
        figure(fig);
    end

    function res = fn_disconnect(dummy)
        fn_DSL_disconnect(echo_on);
        connected = 0;
        res = connected;
    end

%     %set up focal laws
%     function [tx_no, rx_no] = fn_DSL_define_fmc(tx_ch, rx_ch, echo_on)
% 
%     transmit_laws = size(tx_ch, 1);
%     time_traces = length(find(rx_ch));
%     tx_no=zeros(time_traces,1)';
%     rx_no=zeros(time_traces,1)';
%     counter = 0;
% 
%     for fl_ii = 1:transmit_laws %loop through focal laws
% 
%         %clear existing tx delays
% %         for tx_ii = 1:transmit_laws %loop through focal laws
% %             fn_ag_send_command(sprintf('TXF %i %i -1', fl_ii, tx_ii), 0, echo_on);%law, ch, del
% %         end;
%         %find transmitters for each focal law (i.e each row of the tx or rx_matrix
%         tx_nos = find(tx_ch(fl_ii,:));
% %         for tx_ii = 1:length(tx_nos) %add each transmitter specified for focal law
% %             fn_ag_send_command(sprintf('TXF %i %i 0', fl_ii, tx_nos(tx_ii)), 0, echo_on);%law, ch, del
% %         end;
%         %clear existing rx delays
% %         fn_ag_send_command(sprintf('RXF %i 0 -1 0', fl_ii), 0, echo_on);%law, ch, del
%         rx_nos = find(rx_ch(fl_ii,:));
%         for rx_ii = 1:length(rx_nos); %add receivers to all focal laws
%             counter = counter + 1;
% %             fn_ag_send_command(sprintf('RXF %i %i 0 0', fl_ii, rx_nos(rx_ii)), 0, echo_on);%law, ch, del, trim_amp
% 
%             if length(tx_nos)>1
%                 tx_no(counter) = 1;
%             else
%                 tx_no(counter) = tx_nos;
%             end
% 
%             rx_no(counter) = rx_nos(rx_ii);
%         end;
%         %assign focal laws to tests starting at 256
% %         fn_ag_send_command(sprintf('TXN %i %i', 255 + fl_ii, fl_ii), 0, echo_on);
% %         fn_ag_send_command(sprintf('RXN %i %i', 255 + fl_ii, fl_ii), 0, echo_on);
%     end;
%     end

end
