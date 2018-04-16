function available_instruments = fn_get_available_instruments(pt)
bits = sscanf(computer('arch'), 'win%i');
if isdeployed
    available_instruments(1).fn_instrument = @fn_emulator_wrapper;
    if bits == 64
        available_instruments(2).fn_instrument = @fn_tcpip_micropulse_wrapper;
    else
        available_instruments(2).fn_instrument = @fn_tcpip_micropulse_wrapper;
        available_instruments(3).fn_instrument = @fn_micropulse_wrapper;
    end
else
    addpath(genpath(pt));
    %look for wrapper files in instruments folder
    tmp = dir(fullfile(pt, 'fn_*_wrapper.m'));
    if isempty(tmp)
        available_instruments = [];
        return;
    end
    for ii = 1:length(tmp)
        available_instruments(ii).fn_instrument = str2func(tmp(ii).name(1:end-2));
    end
end
fprintf('Available instruments\n');
for ii = 1:length(available_instruments)
    [available_instruments(ii).instr_info, ...
        available_instruments(ii).fn_instr_connect, ...
        available_instruments(ii).fn_instr_disconnect, ...
        available_instruments(ii).fn_instr_reset, ...
        available_instruments(ii).fn_instr_acquire, ...
        available_instruments(ii).fn_send_instr_options] = ...
        available_instruments(ii).fn_instrument();
    fprintf(['  %2i. ', available_instruments(ii).instr_info.name, '\n'], ii);
end

end