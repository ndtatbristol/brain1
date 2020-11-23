function [tfms] = fn_calculate_TFM_for_all_focal_laws(filtered_data,exp_data,focal_laws,use_gpu,varargin)

if (length(varargin)>0)
    interpolation_opt=varargin{1};
    switch interpolation_opt
        case 'nearest'
            interpolation_opt=0;
        case 'linear'
            interpolation_opt=1;
        case 'lanczos2'
            interpolation_opt=2;
        case 'lanczos3'
            interpolation_opt=3;
        case 0 %Nearest
        case 1 %Linear
        case 2 %Lanczos (a=2)
        case 3 %Lanczos (a=3)
        otherwise
            disp('Unknown interpolation option specified for TFM. Using linear')
            interpolation_opt=1;
    end
else
    interpolation_opt=0;
end

if (~isfield(focal_laws,'combined'))
    npixels=size(focal_laws.path,2);
else
	npixels=size(focal_laws.path_tx,1); 
end

tfms.name=focal_laws.name;

exp_data.rx=int32(exp_data.rx);
exp_data.tx=int32(exp_data.tx);

tfms.data=single(zeros(npixels,focal_laws.count));
exp_data.time_data=single(filtered_data);
nelements=length(exp_data.array.el_xc);
nscanlines=length(exp_data.tx);
TFM_focal_law.filter_on = 0;
TFM_focal_law.hilbert_on = 0;

if (~isfield(focal_laws,'combined'))
    npaths=size(focal_laws.path,3);
    focal_laws.path=single(focal_laws.path);
    path_temp=zeros(npixels,nelements,npaths);
    for i=1:npaths
        path_temp(:,:,i)=reshape(focal_laws.path(:,:,i),nelements,npixels).';
    end
end

if (interpolation_opt == 2)
    TFM_focal_law.interpolation_method = 'lanczos2';
elseif (interpolation_opt == 3)
    TFM_focal_law.interpolation_method = 'lanczos3';
elseif (interpolation_opt == 1)
    TFM_focal_law.interpolation_method = 'linear';
else
    TFM_focal_law.interpolation_method = 'nearest';
    t0=exp_data.time(1);
    dt=exp_data.time(2)-exp_data.time(1);
    if (~isfield(focal_laws,'combined'))
        path_temp_ind=int32((path_temp-t0/2)/dt+0.5);
    end
end
TFM_focal_law.lookup_amp = single(ones(npixels, 1,nelements));
TFM_focal_law.lookup_amp_tx = single(ones(npixels, 1,nelements));
TFM_focal_law.lookup_amp_rx = single(ones(npixels, 1,nelements));
TFM_focal_law.filter = single(ones(size(exp_data.time_data,1),1));
    
if (nscanlines == (nelements*nelements+nelements)/2)
    %disp('HMC')
    tt_weight = 2*ones(1,nscanlines);
    tt_weight(exp_data.tx == exp_data.rx)=1;
    TFM_focal_law.hmc_data=true;
else
    %TFM_focal_law=rmfield(TFM_focal_law,'hmc_data');
    tt_weight = ones(1,nscanlines);
end
tt_weight=single(tt_weight);
tt_weight2 =single(ones(1,nscanlines));
exp_data.time_data=single(filtered_data);
for iview=1:focal_laws.count
    itx=focal_laws.raypaths(1,iview);
    irx=focal_laws.raypaths(2,iview);
    if (itx == irx)
        TFM_focal_law.lookup_time=path_temp(:,:,itx);
        TFM_focal_law.lookup_time=reshape(TFM_focal_law.lookup_time,1,npixels,nelements);
        TFM_focal_law.tt_weight=tt_weight;
        if (interpolation_opt == 0)
            TFM_focal_law.lookup_ind=path_temp_ind(:,:,itx);
        end
    else
        if (isfield(TFM_focal_law,'lookup_time'))
            TFM_focal_law=rmfield(TFM_focal_law,'lookup_time');
        end
        if (nscanlines == (nelements*nelements+nelements)/2)
            TFM_focal_law.tt_weight=tt_weight;
        else
            TFM_focal_law.tt_weight=tt_weight2;
        end
        if (~isfield(focal_laws,'combined'))
            TFM_focal_law.lookup_time_tx=path_temp(:,:,itx);
            TFM_focal_law.lookup_time_tx=reshape(TFM_focal_law.lookup_time_tx,1,npixels,nelements);
            TFM_focal_law.lookup_time_rx=path_temp(:,:,irx);
            TFM_focal_law.lookup_time_rx=reshape(TFM_focal_law.lookup_time_rx,1,npixels,nelements);  
            if (interpolation_opt == 0)
                TFM_focal_law.lookup_ind_tx=path_temp_ind(:,:,itx);
                TFM_focal_law.lookup_ind_rx=path_temp_ind(:,:,irx);
            end
        else
            TFM_focal_law.lookup_time_tx=focal_laws.path_tx;
            TFM_focal_law.lookup_time_tx=reshape(TFM_focal_law.lookup_time_tx,1,npixels,nelements);
            TFM_focal_law.lookup_time_rx=focal_laws.path_rx;
            TFM_focal_law.lookup_time_rx=reshape(TFM_focal_law.lookup_time_rx,1,npixels,nelements);  
            if (interpolation_opt == 0)
                TFM_focal_law.lookup_ind_tx=int32((TFM_focal_law.lookup_time_tx-t0/2)/dt+0.5);
                TFM_focal_law.lookup_ind_rx=int32((TFM_focal_law.lookup_time_rx-t0/2)/dt+0.5);
            end
        end
    end
    if (~isfield(focal_laws,'combined'))
        tfms.data(:,iview) = reshape(fn_fast_DAS2(exp_data, TFM_focal_law,use_gpu),npixels,1);
    else
        tfms.data(:,iview) = reshape(gather(fn_fast_DAS3(exp_data, TFM_focal_law,use_gpu)),npixels,1);
    end
end

if (~isfield(focal_laws,'combined'))
    tfms.count=focal_laws.count;
else
    %disp('Reshaping TFM')
    tfms.norm_data=tfms.data(focal_laws.pixels*focal_laws.views+1:end);
    tfms.data=reshape(tfms.data(1:focal_laws.pixels*focal_laws.views),focal_laws.pixels,focal_laws.views);
    tfms.count=focal_laws.views;
end
    
end

