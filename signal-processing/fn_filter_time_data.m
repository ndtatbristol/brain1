function dexp2 = fn_filter_time_data(filter_opt,exp_data,varargin)

% Filter Option
% -1 -> NO FILTER
% = 0 -> Hilbert Filter
% = 1 -> ifft(gaussian * fft(time))
% = 2 -> Butterworth bandpass filter
centre_freq=exp_data.array.centre_freq;
if (length(varargin)>0)
    hb_ratio=varargin{1};
    if (length(varargin)>1)
        centre_freq=varargin{2};
    end  
else
    hb_ratio=0.9;
end

switch filter_opt
    case -1
        dexp2=exp_data.time_data;
    case 0
        dexp2=abs(hilbert(exp_data.time_data));
    case 1
        half_bandwidth=hb_ratio*centre_freq;
        filter = fn_calc_filter(exp_data.time, centre_freq, half_bandwidth,40.0,1);
        dexp2=ifft(spdiags(filter, 0, length(exp_data.time), length(exp_data.time)) * fft(exp_data.time_data));
    case 2
        half_bandwidth=hb_ratio*centre_freq;
        dexp2=fn_butterworth_bandpass(exp_data.time, centre_freq, half_bandwidth,exp_data.time_data);
end

end