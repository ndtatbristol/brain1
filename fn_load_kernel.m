function [focal_law]=fn_load_kernel(focal_law);

if(exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0)
    if exist('fn_fast_DAS3','file')
        if ~isfield(focal_law, 'thread_size')
            focal_law.thread_size=128;
        end
        
        bit_ver=mexext;
        ptx_file=['gpu_tfm' bit_ver([end-1:end]) '.ptx'];
        
        if isfield(focal_law, 'lookup_time')
            sep_tx_rx_laws = 0;
        else
            sep_tx_rx_laws = 1;
        end;
        
        if ~isfield(focal_law, 'hmc_data')
            hmc_data = any(focal_law.tt_weight == 2); %HMC needs to be considered differently if sep tx and rx laws are used
        else
            hmc_data = focal_law.hmc_data;
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
        
        switch method
        case 'Same Tx and RX laws (nearest)'
            focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_near_norm');
            focal_law.lookup_ind=gpuArray(int32(focal_law.lookup_ind));
            focal_law.lookup_amp=gpuArray(single(focal_law.lookup_amp));
        case 'Different Tx and RX laws, FMC data (nearest)'
            focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_near_2dly');
            focal_law.lookup_ind_tx=gpuArray(int32(focal_law.lookup_ind_tx));
            focal_law.lookup_ind_rx=gpuArray(int32(focal_law.lookup_ind_rx));
            focal_law.lookup_amp_tx=gpuArray(single(focal_law.lookup_amp_tx));
            focal_law.lookup_amp_rx=gpuArray(single(focal_law.lookup_amp_rx));
        case 'Different Tx and RX laws, HMC data (nearest)'
            focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_near_hmc');
            focal_law.lookup_ind_tx=gpuArray(int32(focal_law.lookup_ind_tx));
            focal_law.lookup_ind_rx=gpuArray(int32(focal_law.lookup_ind_rx));
            focal_law.lookup_amp_tx=gpuArray(single(focal_law.lookup_amp_tx));
            focal_law.lookup_amp_rx=gpuArray(single(focal_law.lookup_amp_rx));
        case 'Same Tx and RX laws (linear)'
            focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_norm');
            focal_law.lookup_time=gpuArray(single(focal_law.lookup_time));
            focal_law.lookup_amp=gpuArray(single(focal_law.lookup_amp));
        case 'Different Tx and RX laws, FMC data (linear)'
            focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_2dly');
            focal_law.lookup_time_tx=gpuArray(single(focal_law.lookup_time_tx));
            focal_law.lookup_time_rx=gpuArray(single(focal_law.lookup_time_rx));
            focal_law.lookup_amp_tx=gpuArray(single(focal_law.lookup_amp_tx));
            focal_law.lookup_amp_rx=gpuArray(single(focal_law.lookup_amp_rx));
        case 'Different Tx and RX laws, HMC data (linear)'
            focal_law.kern = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_hmc');
            focal_law.lookup_time_tx=gpuArray(single(focal_law.lookup_time_tx));
            focal_law.lookup_time_rx=gpuArray(single(focal_law.lookup_time_rx));
            focal_law.lookup_amp_tx=gpuArray(single(focal_law.lookup_amp_tx));
            focal_law.lookup_amp_rx=gpuArray(single(focal_law.lookup_amp_rx));
    end
    
    focal_law.kern.ThreadBlockSize = focal_law.thread_size;
        
    end
end