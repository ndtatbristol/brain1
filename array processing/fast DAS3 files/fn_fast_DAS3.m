function [result, varargout] = fn_fast_DAS3(exp_data, focal_law,varargin)
%USAGE
%   [result, [timings]] = fn_fast_DAS3(exp_data, focal_law,[use_gpu_if_available])
%SUMMARY
%   Delay and sum implementation that can be used with any linear imaging
%   algorithm where tx and rx weights are frequency independent. Can be
%   used either with same focal law on transmit and receive (in which case
%   focal_law.lookup_time, focal_law.lookup_ind and focal_law.lookup_amp
%   must exist) or with separate transmit and receive focal laws (in which
%   case focal_law.lookup_time_tx, focal_law.lookup_time_rx etc. must
%   exist).
%INPUTS
%   exp_data - experimental data in usual format
%   focal_law - structured variable that effectively contains imaging
%   algorithm (described below for same tx and rx focal law - if different
%   laws are required then two sets of following are needed suffixed with
%   _tx and _rx).
%   use_gpu_if_available [1] - set to one to use the CPU image calculation. By default if
%   available fn_fast_DAS2 will use the GPU.
%   focal_law.lookup_time - 3D matrix of propagation times from each element
%   to each image point
%   focal_law.lookup_ind - 3D matrix of equivalent indices (round(t/dt)) from
%   each element to each image point
%   focal_law.lookup_amp - 3D matrix of weightings associated with each
%   element and each image point
%   focal_law.hilbert_on - flag to do hilbert transform of each signal prior
%   to processing
%   focal_law.filter_on - flag to include pre-filtering of time-traces
%   focal_law.filter - frequency domain filter to apply to time-traces if
%   required
%   focal_law.tt_ind - optional indices of each time-trace to process (if not
%   specified, they will all be processed, regardless of whether or not
%   they make any contributions to image, so using this vector can save
%   time with algorithms such as b-scans where certain time-traces in FMC
%   data are not used at all)
%   focal_law.tt_weight - weighting associated with each time trace (used to
%   double pitch-catch signals if half matrix capture used)
%   focal_law.interpolation_method = sets how time-domain data is
%   interpolated; 'nearest' means no interpolation and nearest point is
%   used; 'linear' means linear interpolation
%   [use_gpu_if_available = 1] - optional argument to enable/ disable use of GPU if
%   one is present. Default is enabled.
%OUTPUTS
%   result - the resulting image
%   timings - if specified returns times taken for different steps of
%   calculation for debugging/comparison purposes.

%--------------------------------------------------------------------------
global SYSTEM_TYPES
if ~isfield(SYSTEM_TYPES,'mat_ver')
    a=ver('matlab');
    SYSTEM_TYPES.mat_ver=str2num(a.Version);
end

tic
if nargout > 1
    record_times = 1;
else
    record_times = 0;
end

if nargin > 2
    use_gpu_if_available = varargin{1};
else
    use_gpu_if_available = 1;
end

if use_gpu_if_available && (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0)
    if ~isfield(focal_law, 'tt_ind')
        focal_law.tt_ind = 1:length(exp_data.tx);
    end
    
    if ~isfield(focal_law, 'thread_size')
%         gpu_han=gpuDevice(1);
        focal_law.thread_size=128; %hard-coded nightmare! should be OK
    end
    
    if ~isfield(focal_law, 'filter')
        focal_law.filter=ones(size(exp_data.time));
    end
    
    if isfield(focal_law, 'lookup_time')
        sep_tx_rx_laws = 0;
        img_size=size(focal_law.lookup_time);
    else
        sep_tx_rx_laws = 1;
        img_size=size(focal_law.lookup_time_tx);
    end;
    
    
    if ~isfield(focal_law, 'hmc_data')
        hmc_data = any(focal_law.tt_weight == 2); %HMC needs to be considered differently if sep tx and rx laws are used
    else
        hmc_data = focal_law.hmc_data;
    end
    
    if record_times
        varargout{1}.filter = double(toc);
    end
    
    if sep_tx_rx_laws
        if hmc_data
            method = 'Different Tx and RX laws, HMC data';
        else
            method = 'Different Tx and RX laws, FMC data';
        end
    else
        method = 'Same Tx and RX laws';
    end
    
    method = [method, ' (', lower(focal_law.interpolation_method), ')'];
    
    %define the area of the image field
    ndims=length(img_size);
    img_size=img_size(1:end-1);
    %result=zeros(img_size);
    
    %convert data to required gpuarray data
    %result=parallel.gpu.GPUArray.zeros(size(result),'single');
    n=gpuArray( int32(size(exp_data.time_data,1)));
    combs=gpuArray( int32(size(exp_data.time_data,2)));
    tx=gpuArray( int32(exp_data.tx));
    rx=gpuArray( int32(exp_data.rx));
    tt_weight = gpuArray(single(focal_law.tt_weight));
    %filter_weight = parallel.gpu.GPUArray.ones(size(focal_law.tt_weight),'single');
    time_data=gpuArray(single(exp_data.time_data));
    
    if SYSTEM_TYPES.mat_ver>=8
        result=gpuArray.zeros(img_size,'single');
        filter_weight = gpuArray.ones(size(focal_law.tt_weight),'single');
        first_half_filt=gpuArray.ones(floor(single(n)/2),1,'single');
        scnd_half_filt=gpuArray.zeros(ceil(single(n)/2),1,'single');
    else
        result=parallel.gpu.GPUArray.ones(img_size,'single');
        filter_weight = parallel.gpu.GPUArray.ones(size(focal_law.tt_weight),'single');
        first_half_filt=parallel.gpu.GPUArray.ones(floor(single(n)/2),1,'single');
        scnd_half_filt=parallel.gpu.GPUArray.zeros(ceil(single(n)/2),1,'single');
    end
    
    if focal_law.filter_on
        filter=gpuArray(single(focal_law.filter));
        time_data=ifft((filter*tt_weight.*fft(time_data)))*2;
    else
        if focal_law.hilbert_on
            %first_half_filt=parallel.gpu.GPUArray.ones(floor(single(n)/2),1,'single');
            %scnd_half_filt=parallel.gpu.GPUArray.zeros(ceil(single(n)/2),1,'single');
            filter=[first_half_filt; scnd_half_filt];
            time_data=ifft((filter*filter_weight.*fft(time_data)))*2;
        end
    end
    
    %split analytic data into real and imaginary parts
    real_exp=single(real(time_data));
    img_exp=single(imag(time_data));
    pixs=prod(img_size);
    
    %specify total number of pixels and the image dimensions
    pixels=gpuArray(int32(pixs));
    grid_x=gpuArray(int32(size(result,1)));
    grid_y=gpuArray(int32(size(result,2)));
    grid_z=gpuArray(int32(size(result,3)));
    real_result=result;
    imag_result=result;
    
    bit_ver=mexext;
    ptx_file=['gpu_tfm' bit_ver([end-1:end]) '.ptx'];
    
    %Now the main loops - different loop depending on method and whether a kernel is preloaded.
    switch method
        case 'Same Tx and RX laws (nearest)'
            if ~isfield(focal_law, 'kern')
                focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_near_norm');
                focal_law.kern.ThreadBlockSize = 128;
                focal_law.lookup_ind= gpuArray(int32(focal_law.lookup_ind));
                focal_law.lookup_amp= gpuArray(single(focal_law.lookup_amp));
            end
            focal_law.kern.GridSize = ceil(pixs./focal_law.kern.ThreadBlockSize(1));
            
            [real_result imag_result]=feval(focal_law.kern,real_result,imag_result,n,combs,real_exp,img_exp,tx, rx, focal_law.lookup_ind, pixels, grid_x, grid_y, grid_z, focal_law.lookup_amp, tt_weight);
        case 'Different Tx and RX laws, FMC data (nearest)'
            if ~isfield(focal_law, 'kern')
                focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_near_2dly');
                focal_law.kern.ThreadBlockSize = 128;
                focal_law.lookup_ind_tx= gpuArray(int32(focal_law.lookup_ind_tx));
                focal_law.lookup_ind_rx= gpuArray(int32(focal_law.lookup_ind_rx));
                focal_law.lookup_amp_tx=gpuArray(single(focal_law.lookup_amp_tx));
                focal_law.lookup_amp_rx=gpuArray(single(focal_law.lookup_amp_rx));
            end
            focal_law.kern.GridSize = ceil(pixs./focal_law.kern.ThreadBlockSize(1));
            
            [real_result imag_result]=feval(focal_law.kern,real_result,imag_result,n,combs,real_exp,img_exp,tx, rx, focal_law.lookup_ind_tx, focal_law.lookup_ind_rx, pixels, grid_x, grid_y, grid_z, focal_law.lookup_amp_tx,focal_law.lookup_amp_rx,tt_weight);
        case 'Different Tx and RX laws, HMC data (nearest)'
            if ~isfield(focal_law, 'kern')
                focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_near_hmc');
                focal_law.kern.ThreadBlockSize = 128;
                focal_law.lookup_ind_tx= gpuArray(int32(focal_law.lookup_ind_tx));
                focal_law.lookup_ind_rx= gpuArray(int32(focal_law.lookup_ind_rx));
                focal_law.lookup_amp_tx=gpuArray(single(focal_law.lookup_amp_tx));
                focal_law.lookup_amp_rx=gpuArray(single(focal_law.lookup_amp_rx));
            end
            focal_law.kern.GridSize = ceil(pixs./focal_law.kern.ThreadBlockSize(1));
            
            [real_result imag_result]=feval(focal_law.kern,real_result,imag_result,n,combs,real_exp,img_exp,tx, rx, focal_law.lookup_ind_tx, focal_law.lookup_ind_rx, pixels, grid_x, grid_y, grid_z, focal_law.lookup_amp_tx,focal_law.lookup_amp_rx,tt_weight);
            
            %Linear interpolation methods
            
        case 'Same Tx and RX laws (linear)'
            if ~isfield(focal_law, 'kern')
                focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_norm');
                focal_law.kern.ThreadBlockSize = 128;
                %focal_law.lookup_time= gpuArray(single(focal_law.lookup_time));
                %focal_law.lookup_amp= gpuArray(single(focal_law.lookup_amp));
            end
            focal_law.kern.GridSize = ceil(pixs./focal_law.kern.ThreadBlockSize(1));
            
            time= gpuArray(single(exp_data.time));
                                 
            [real_result imag_result]=feval(focal_law.kern,real_result,imag_result,n,combs,real_exp,img_exp,tx, rx, focal_law.lookup_time,time, pixels, grid_x, grid_y, grid_z, focal_law.lookup_amp, tt_weight);
            
        case 'Different Tx and RX laws, FMC data (linear)'
            if ~isfield(focal_law, 'kern')
                focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_2dly');
                focal_law.kern.ThreadBlockSize = 128;
                focal_law.lookup_time_tx= gpuArray(single(focal_law.lookup_time_tx));
                focal_law.lookup_time_rx= gpuArray(single(focal_law.lookup_time_rx));
                focal_law.lookup_amp_tx= gpuArray(single(focal_law.lookup_amp_tx));
                focal_law.lookup_amp_rx= gpuArray(single(focal_law.lookup_amp_rx));
            end
            focal_law.kern.GridSize = ceil(pixs./focal_law.kern.ThreadBlockSize(1));
            
            time= gpuArray(single(exp_data.time));
            
            [real_result imag_result]=feval(focal_law.kern,real_result,imag_result,n,combs,real_exp,img_exp,tx, rx, focal_law.lookup_time_tx, focal_law.lookup_time_rx,time, pixels, grid_x, grid_y, grid_z, focal_law.lookup_amp_tx, focal_law.lookup_amp_rx, tt_weight);
            
        case 'Different Tx and RX laws, HMC data (linear)'
            if ~isfield(focal_law, 'kern')
                focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_hmc');
                focal_law.kern.ThreadBlockSize = 128;
                focal_law.lookup_time_tx= gpuArray(single(focal_law.lookup_time_tx));
                focal_law.lookup_time_rx= gpuArray(single(focal_law.lookup_time_rx));
                focal_law.lookup_amp_tx= gpuArray(single(focal_law.lookup_amp_tx));
                focal_law.lookup_amp_rx= gpuArray(single(focal_law.lookup_amp_rx));
            end
            focal_law.kern.GridSize = ceil(pixs./focal_law.kern.ThreadBlockSize(1));
            
            time= gpuArray(single(exp_data.time));
            
            [real_result imag_result]=feval(focal_law.kern,real_result,imag_result,n,combs,real_exp,img_exp,tx, rx, focal_law.lookup_time_tx, focal_law.lookup_time_rx,time, pixels, grid_x, grid_y, grid_z, focal_law.lookup_amp_tx, focal_law.lookup_amp_rx, tt_weight);
            
    end
    result=real_result+imag_result.*1i;
    if length(img_size)==3
        result=gather(result);
        result=double(result);
    end
else
    
    if isfield(focal_law, 'lookup_time')
        result_dims = size(focal_law.lookup_time);
    else
        if prod(size(focal_law.lookup_time_tx)) > prod(size(focal_law.lookup_time_rx))
            result_dims = size(focal_law.lookup_time_tx);
        else
            result_dims = size(focal_law.lookup_time_rx);
        end
    end
    
    result_dims = result_dims(1:end - 1);
    result = zeros(result_dims);
    result = result(:);%vectorise
    
    if ~isfield(focal_law, 'tt_ind')
        focal_law.tt_ind = 1:length(exp_data.tx);
    end
    
    if isfield(focal_law, 'lookup_time')
        sep_tx_rx_laws = 0;
        focal_law.lookup_time = reshape(focal_law.lookup_time, prod(result_dims), []);
        focal_law.lookup_ind = reshape(focal_law.lookup_ind, prod(result_dims), []);
        focal_law.lookup_amp = reshape(focal_law.lookup_amp, prod(result_dims), []);
    else
        sep_tx_rx_laws = 1;
        focal_law.lookup_time_tx = reshape(focal_law.lookup_time_tx, prod(result_dims), []);
        focal_law.lookup_ind_tx = reshape(focal_law.lookup_ind_tx, prod(result_dims), []);
        focal_law.lookup_amp_tx = reshape(focal_law.lookup_amp_tx, prod(result_dims), []);
        focal_law.lookup_time_rx = reshape(focal_law.lookup_time_rx, prod(result_dims), []);
        focal_law.lookup_ind_rx = reshape(focal_law.lookup_ind_rx, prod(result_dims), []);
        focal_law.lookup_amp_rx = reshape(focal_law.lookup_amp_rx, prod(result_dims), []);
    end;
    
    % if ~focal_law.amp_on
    %     if sep_tx_rx_laws
    %         focal_law.lookup_amp_tx = ones(size(focal_law.lookup_amp_tx));
    %         focal_law.lookup_amp_rx = ones(size(focal_law.lookup_amp_rx));
    %     else
    %         focal_law.lookup_amp = ones(size(focal_law.lookup_amp));
    %     end
    % end
    
    %see if data is HMC unless explicitly stated by looking for and tt_weight
    %values of 2
    if ~isfield(focal_law, 'hmc_data')
        hmc_data = any(focal_law.tt_weight == 2); %HMC needs to be considered differently if sep tx and rx laws are used
    else
        hmc_data = focal_law.hmc_data;
    end
    
    %filtering
    if focal_law.filter_on
        exp_data.time_data = ifft(spdiags(focal_law.filter, 0, length(exp_data.time), length(exp_data.time)) * fft(exp_data.time_data));
    else
        if focal_law.hilbert_on
            exp_data.time_data = ifft(spdiags([1:length(exp_data.time)]' < length(exp_data.time) / 2, 0, length(exp_data.time), length(exp_data.time)) * fft(exp_data.time_data));
        end
    end
    
    %add line of zeros to tail of data so that out of range values are zero
    %in interpolation or lookup stage
    
    exp_data.time_data = [exp_data.time_data; zeros(1, size(exp_data.time_data, 2))];
    n = size(exp_data.time_data, 1);
    
    if record_times
        varargout{1}.filter = double(toc);
    end
    
    if sep_tx_rx_laws
        if hmc_data
            method = 'Different Tx and RX laws, HMC data';
        else
            method = 'Different Tx and RX laws, FMC data';
        end
    else
        method = 'Same Tx and RX laws';
    end
    
    method = [method, ' (', focal_law.interpolation_method, ')'];
    
    %Now the main loops - different loop depending on method.
    switch method
        case 'Same Tx and RX laws (nearest)'
            
            for ii = 1:length(focal_law.tt_ind)
                jj = focal_law.tt_ind(ii);
                amp1 = ...
                    focal_law.lookup_amp(:, exp_data.tx(jj)) .* ...
                    focal_law.lookup_amp(:, exp_data.rx(jj)) * ...
                    focal_law.tt_weight(jj);
                ind1 = ...
                    focal_law.lookup_ind(:, exp_data.tx(jj)) + ...
                    focal_law.lookup_ind(:, exp_data.rx(jj));
                %out of range indices set to n (to give a zero value in result)
                ind1(find(ind1 < 1)) = n;
                ind1(find(ind1 > n)) = n;
                result = result + exp_data.time_data(ind1, jj) .* amp1(:);
            end
            
        case 'Different Tx and RX laws, FMC data (nearest)'
            %Separate focal laws for Tx and Rx and FMC data
            for ii = 1:length(focal_law.tt_ind)
                jj = focal_law.tt_ind(ii);
                amp1 = ...
                    focal_law.lookup_amp_tx(:, exp_data.tx(jj)) .* ...
                    focal_law.lookup_amp_rx(:, exp_data.rx(jj)) * ...
                    focal_law.tt_weight(jj);
                ind1 = ...
                    focal_law.lookup_ind_tx(:, exp_data.tx(jj)) + ...
                    focal_law.lookup_ind_rx(:, exp_data.rx(jj));
                %out of range indices set to n (to give a zero value in result)
                ind1(find(ind1 < 1)) = n;
                ind1(find(ind1 > n)) = n;
                result = result + exp_data.time_data(ind1, jj) .* amp1(:);
            end
            
        case 'Different Tx and RX laws, HMC data (nearest)'
            
            for ii = 1:length(focal_law.tt_ind)
                jj = focal_law.tt_ind(ii);
                ind1 = ...
                    focal_law.lookup_ind_tx(:, exp_data.tx(jj)) + ...
                    focal_law.lookup_ind_rx(:, exp_data.rx(jj));
                amp1 = ...
                    focal_law.lookup_amp_tx(:, exp_data.tx(jj)) .* ...
                    focal_law.lookup_amp_rx(:, exp_data.rx(jj)) * ...
                    focal_law.tt_weight(jj) / 2;
                ind2 = ...
                    focal_law.lookup_ind_tx(:, exp_data.rx(jj)) + ...
                    focal_law.lookup_ind_rx(:, exp_data.tx(jj));
                amp2 = ...
                    focal_law.lookup_amp_tx(:, exp_data.rx(jj)) .* ...
                    focal_law.lookup_amp_rx(:, exp_data.tx(jj)) * ...
                    focal_law.tt_weight(jj) / 2;
                %out of range indices set to n (to give a zero value in result)
                ind1(find(ind1 < 1)) = n;
                ind1(find(ind1 > n)) = n;
                ind2(find(ind2 < 1)) = n;
                ind2(find(ind2 > n)) = n;
                result = result + ...
                    exp_data.time_data(ind1, jj) .* amp1 + ...
                    exp_data.time_data(ind2, jj) .* amp2;
            end
            
            %Linear interpolation methods
            
        case 'Same Tx and RX laws (linear)'
            
            for ii = 1:length(focal_law.tt_ind)
                jj = focal_law.tt_ind(ii);
                amp1 = ...
                    focal_law.lookup_amp(:, exp_data.tx(jj)) .* ...
                    focal_law.lookup_amp(:, exp_data.rx(jj)) * ...
                    focal_law.tt_weight(jj);
                time1 = ...
                    focal_law.lookup_time(:, exp_data.tx(jj)) + ...
                    focal_law.lookup_time(:, exp_data.rx(jj));
                result = result + ...
                    fn_fast_linear_interp(exp_data.time, exp_data.time_data(:, jj), time1(:)) .* amp1(:);
            end
            
        case 'Different Tx and RX laws, FMC data (linear)'
            %Separate focal laws for Tx and Rx and FMC data
            for ii = 1:length(focal_law.tt_ind)
                jj = focal_law.tt_ind(ii);
                amp1 = ...
                    focal_law.lookup_amp_tx(:, exp_data.tx(jj)) .* ...
                    focal_law.lookup_amp_rx(:, exp_data.rx(jj)) * ...
                    focal_law.tt_weight(jj);
                time1 = ...
                    focal_law.lookup_time_tx(:, exp_data.tx(jj)) + ...
                    focal_law.lookup_time_rx(:, exp_data.rx(jj));
                result = result + ...
                    fn_fast_linear_interp(exp_data.time, exp_data.time_data(:, jj), time1(:)) .* amp1(:);
            end
            
        case 'Different Tx and RX laws, HMC data (linear)'
            
            for ii = 1:length(focal_law.tt_ind)
                jj = focal_law.tt_ind(ii);
                time1 = ...
                    focal_law.lookup_time_tx(:, exp_data.tx(jj)) + ...
                    focal_law.lookup_time_rx(:, exp_data.rx(jj));
                amp1 = ...
                    focal_law.lookup_amp_tx(:, exp_data.tx(jj)) .* ...
                    focal_law.lookup_amp_rx(:, exp_data.rx(jj)) * ...
                    focal_law.tt_weight(jj) / 2;
                time2 = ...
                    focal_law.lookup_time_tx(:, exp_data.rx(jj)) + ...
                    focal_law.lookup_time_rx(:, exp_data.tx(jj));
                amp2 = ...
                    focal_law.lookup_amp_tx(:, exp_data.rx(jj)) .* ...
                    focal_law.lookup_amp_rx(:, exp_data.tx(jj)) * ...
                    focal_law.tt_weight(jj) / 2;
                result = result + ...
                    fn_fast_linear_interp(exp_data.time, exp_data.time_data(:, jj), time1(:)) .* amp1(:) + ...
                    fn_fast_linear_interp(exp_data.time, exp_data.time_data(:, jj), time2(:)) .* amp2(:);
            end
            
    end
    
    %finally reshape image back to desired dimensions
    result = reshape(result, result_dims);%back to right shape
end
if record_times
    varargout{1}.loop = double(toc) - varargout{1}.filter;
    varargout{1}.total = double(toc);
end
end

function yi = fn_fast_linear_interp(x, y, xi)
%note out of range xi values return zero
x = x(:);
y = y(:);
xi = xi(:);
dx = x(2) - x(1);
x0 = x(1);
ii = floor((xi - x0) / dx) + 1;
j1 = find(ii < 1);
ii(j1) = 1;
j2 = find(ii > (length(x) - 1));
ii(j2) = length(x) - 1;
yi = y(ii) + (y(ii + 1) - y(ii)) .* (xi - x(ii)) ./ dx;
yi(j1) = 0;
yi(j2) = 0;
end