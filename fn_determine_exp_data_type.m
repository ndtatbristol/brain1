function t = fn_determine_exp_data_type(exp_data)
%SUMMARY
%   Determines type (HMC, FMC, CSM, SAFT) of array data

%--------------------------------------------------------------------------
no_tx = length(unique(exp_data.tx));
no_rx = length(unique(exp_data.rx));
no_tt = length(exp_data.tx);
if (no_tt == (no_tx * no_rx)) & (no_tx > 1) & (no_rx > 1)
    t = 'FMC';
    return
end
if no_tt == (no_tx * (no_rx + 1) / 2)
    t = 'HMC';
    return
end
if (no_tt == no_tx) & (no_tt == no_rx)
    t = 'SAFT';
    return
end
if (no_tx == 1) & (no_tt == no_rx)
    t = 'CSM';
    return
end
t = 'Unknown';
warning('Unknown array data type');
end