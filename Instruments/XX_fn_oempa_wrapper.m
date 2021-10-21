function [info, h_fn_connect, h_fn_disconnect, h_fn_reset, h_fn_acquire, h_fn_send_options] = fn_oempa_wrapper(dummy)
info.name = 'AOS OEMPA';
echo_on = 0;

% info.options_info.ip_address.label = 'IP address';
% info.options_info.ip_address.default = '10.1.1.2';
% info.options_info.ip_address.type = 'string';
%
% info.options_info.port_no.label = 'Port number';
% info.options_info.port_no.default = 1067;
% info.options_info.port_no.type = 'int';
% info.options_info.port_no.constraint = [1, 99999];

info.options_info.acquire_mode.label = 'Acquisition';
info.options_info.acquire_mode.default = 'FMC'; %Sample frequency (MHz); int; 1e6; {25, 50, 100}
info.options_info.acquire_mode.type = 'constrained';
%info.options_info.acquire_mode.constraint = {'SAFT', 'FMC', 'HMC', 'CSM'};
info.options_info.acquire_mode.constraint = {'FMC'};

info.options_info.sample_freq.label = 'Sample frequency (MHz)';
info.options_info.sample_freq.default = '25'; %Sample frequency (MHz); int; 1e6; {25, 50, 100}
info.options_info.sample_freq.type = 'constrained';
info.options_info.sample_freq.constraint = {'10', '25', '50', '100'};

% info.options_info.pulse_voltage.label = 'Pulse voltage (V)';
% info.options_info.pulse_voltage.default = 100;
% info.options_info.pulse_voltage.type = 'double';
% info.options_info.pulse_voltage.constraint = [50, 300];
% info.options_info.pulse_voltage.multiplier = 1;

info.options_info.pulse_width.label = 'Pulse width (ns)';
info.options_info.pulse_width.default = 100e-9;
info.options_info.pulse_width.type = 'double';
info.options_info.pulse_width.constraint = [20e-9, 500e-9];
info.options_info.pulse_width.multiplier = 1e-9;

info.options_info.time_pts.label = 'Time points';
info.options_info.time_pts.default = 3000;
info.options_info.time_pts.type = 'int'; % why int?
info.options_info.time_pts.constraint = [100, 3000];

info.options_info.sample_bits.label = 'Sample bits';
info.options_info.sample_bits.default = '8';
info.options_info.sample_bits.type = 'constrained';
info.options_info.sample_bits.constraint = {'8','12','16'};

info.options_info.db_gain.label = 'Gain (dB)';
info.options_info.db_gain.default = 20;
info.options_info.db_gain.type = 'int';
info.options_info.db_gain.constraint = [0, 70];%check!

% info.options_info.filter_no.label = 'Filter number';
% info.options_info.filter_no.default = 4;
% info.options_info.filter_no.type = 'int';
% info.options_info.filter_no.constraint = [0, 4];

% info.options_info.prf.label = 'Maximum PRF (kHz)';
% info.options_info.prf.default = 2e3;
% info.options_info.prf.type = 'double';
% info.options_info.prf.multiplier = 1e3;
% info.options_info.prf.constraint = [0.001e3 20e3];

% info.options_info.averages.label = 'Averages';
% info.options_info.averages.default = 1;
% info.options_info.averages.type = 'int';
% info.options_info.averages.constraint = [1, 64];

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
deviceId = [];

    function exp_data = fn_acquire(dummy)
        exp_data = [];
        if ~options_sent
            %this should give a warning!
            return;
        end
        if ~connected
            return;
        end
        % get the cycle count (from setting, value in the OEMPAfile).
        CycleCount = mxGetSWCycleCount(deviceId);
        
        % Enable pulser %
        EnableCall = mxEnableShot(deviceId,1);
        
        tstartTimerDisplay = tic;
        lastDisplay = [];
        [AscanCount fifoAscanLost1 total] = mxGetAcquisitionAscanFifoStatus(deviceId) ;
        while AscanCount<CycleCount;
            [AscanCount fifoAscanLost1 total] = mxGetAcquisitionAscanFifoStatus(deviceId) ;
            [lastDisplay displayOn] = TimerDisplay(tstartTimerDisplay,lastDisplay);
            if displayOn
                fprintf('Loop init %i, lost=%i total=%i\n',AscanCount,fifoAscanLost1,total);
            end
        end
        
        [FifoIndex, Cycle, Sequence, xPointCount, ByteSize, Signed] = mxGetAcquisitionAscanFifoIndex(deviceId,linspace(0,CycleCount-1,CycleCount));
        PointCount = max(xPointCount);
        
        % Check the number of bits %
        if strcmp(mxGetAcquisitionAscanBitSize(deviceId,0),'8Bits')
            Nbit = 8;
        elseif strcmp(mxGetAcquisitionAscanBitSize(deviceId,0),'12Bits')
            Nbit = 12;
        elseif strcmp(mxGetAcquisitionAscanBitSize(deviceId,0),'16Bits')
            Nbit = 16;
        end
        if Nbit>=12
            PointCount = PointCount/2;
        end
        
        Ascan = zeros(PointCount,CycleCount); %storage of all ascan
        
        %-------------------------------------------------------------------------%
        row = 0; col = 0;
        while (row~=PointCount) | (col~=CycleCount)
            [AscanCount fifoAscanLost1 total] = mxGetAcquisitionAscanFifoStatus(deviceId) ;
            [lastDisplay displayOn] = TimerDisplay(tstartTimerDisplay,lastDisplay);
            if displayOn
                fprintf('count %d lost %d total %d\n',AscanCount,fifoAscanLost1,total)
            end
            % Cumulate Ascans %
            FifoIndex = mxGetAcquisitionAscanLifoIndex(deviceId, linspace(0,CycleCount-1,CycleCount));
            [Ascan, Cycle, Sequence, encRawVal, lEncoder] = mxGetAcquisitionAscanFifoData(deviceId, FifoIndex);
            [row, col] = size(Ascan);
        end
        %-------------------------------------------------------------------------%
        
        % Get Ascan lost number %
        NascanLost = mxGetLostCountAscan(deviceId);
        exp_data.time_data=double(Ascan);
        exp_data.time_data=exp_data.time_data(1:length(time_axis),:);
        exp_data.tx = tx_no;
        exp_data.rx = rx_no;
        exp_data.time = time_axis;
        
%         Reset = mxResetCounters(deviceId);

        % Disable pulser %
        EnableCall = mxEnableShot(deviceId,0);
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
%         options.sample_freq  = 100e6;
        options.sample_bits = str2double(options.sample_bits);
        write_files('test.txt',no_channels,options.sample_bits,options.gate_start * 1e6, options.time_pts / str2double(options.sample_freq) , options.db_gain, options.pulse_width * 1e6,str2double(options.sample_freq)*1e6);                                
        tmp = meshgrid([1:no_channels],[1:no_channels]);
        tx_no = reshape(tmp,[1 no_channels.^2]);
        rx_no = repmat([1:no_channels],1,no_channels);
        %         [tx_no, rx_no] = fn_ag_set_test_options_tcpip(options, echo_on);
        time_step = 1 / str2double(options.sample_freq)*1e-6;
        % Load the setup file %
        status = mxReadFileWriteHW(deviceId,'test.txt');
        if ~status
            error('Cannot load the setup file');
            %     break
        end
        time_axis = [options.gate_start:time_step:options.gate_start + time_step*(options.time_pts-1)]' - options.instrument_delay;
        options_sent = 1;
    end

    function fn_reset(dummy)
        fn_ag_reset_tcpip(echo_on);
    end

    function res = fn_disconnect(dummy)
        if ~connected
            return
        end
        %         fn_ag_disconnect_tcpip(echo_on);
        % Disable pulser %
        %EnableCall = mxEnableShot(deviceId,0);
        
        % Disconnect %
        mxConnect(deviceId,0);
        
        % Delete device %
        mxDeleteDevice(deviceId);
        
        % free matlab stub dll but makes Matlab crash %
        utCmdExit
        connected = 0;
        res = connected;
    end

    function res = fn_connect(options)
        % Path to find matlab MEX files %
        addpath('C:\Program Files\AOS\OEMPA 1.1.5.5\matlab');
        
        % Load matlab stub dll %
        utCmdInit('C:\Program Files\AOS\OEMPA 1.1.5.5\UTKernelMatlab.dll');
        
        % New device (specify IP address) %
        deviceId = utCmdNewDevice('192.168.1.11',[16384 5000 1]);
        
        % Connection %;
        status = mxConnect(deviceId,1);
        if ~status
            connected = 0;
            error('Cannot connect to the device');
        else
            connected = 1;
        end
        res = connected;
    end
end


