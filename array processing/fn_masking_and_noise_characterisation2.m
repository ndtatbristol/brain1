function [sigma_c,stored_mask]=fn_masking_and_noise_characterisation2(mesh,tfmdata,varargin)

% Function expects minimum of two inputs

% input 1: structure containing .x and .z, which are 1D arrays containing x
% and z coordinate components. Note: not full 2D mesh (npixels = length(z) * length(x) )
% input 2: TFM data array arranged as (npixels,nviews)
% input 3 (optional): percentile controlling image masking , typically 99th
% input 4 (optional): Max number of pixels to remove per masking iteration
% input 5 (optional): In-fill hole (=0 off, >0 is pixel hole size) after masking
% input 6 (optional): Expand by specified pixels (=0 off, >0 expansion size) after masking
% input 7 (optional): Suppress text output (=0 suppress, otherwise display)

% Output 1: Rayleigh parameter sigma per view
% Output 2: Mask arranged in (npixels,nviews) format

[npixels,nviews]=size(tfmdata);

perc_search=99.0;
nRemovals=0.02*npixels;
hole_fill=0;
hole_expand=0;
text_output=0;
if (length(varargin)>0)
    perc_search=varargin{1};
    if (length(varargin)>1)
        nRemovals=varargin{2};
        if (length(varargin)>2)
            hole_fill=varargin{3};
            if (length(varargin)>3)
                hole_expand=varargin{4};
                if (length(varargin)>4)
                    text_output=varargin{5};
                end
            end
        end
    end
end

tfmdata=abs(tfmdata);


pixel_size=mesh.x(2)-mesh.x(1);

pixel_dilation=10;
view_psf_local=pixel_dilation*pixel_size*ones(length(mesh.z),length(mesh.x));
stored_mask=zeros(size(tfmdata));
sigma_c=zeros(nviews,1);
for iview=1:nviews
    rk_passed=1:npixels;
    latest_mask=zeros(length(mesh.z),length(mesh.x));
    tstart2=tic;
    i=1;
    nRemovals2=nRemovals; ii=0; val=100;
    while (i>0)
        i=i+1;
        if (ii < 1)
            latest_mask_old=latest_mask;
            val_old=val;
        end
        rk_passed=find(latest_mask<1);
        if (length(rk_passed) < 100)
            break;
        end
        IvalsCorrected =tfmdata(rk_passed,iview);
        %IvalsCorrected=Ivals./attenMaps.data(rk_passed,iview); 

        sigma = fn_rayleigh_mle(IvalsCorrected(:)); perc_frac=perc_search/100.0;
        threshold2=sigma*sqrt(-2*log(1-perc_frac));
        rk_passed2=find(IvalsCorrected(:) >= threshold2); rk_passed2=rk_passed(rk_passed2);
        q_array=zeros(size(latest_mask)); q_array(latest_mask>0)=2; q_array(rk_passed2)=1;

        fraction_mask = fn_hics_form_neighbour_mask_fixed(q_array,view_psf_local,1.0,pixel_size);
        
        for iRemovals=1:nRemovals2
            [max1,max2]=max(fraction_mask);
            [max1b,ixx0]=max(max1);
             if (max1b < 1.0e-9)
                 break;
             end
            izz0=max2(ixx0);
            %find largest cluster
             overlap0=sub2ind(size(latest_mask),izz0,ixx0);
             if (iRemovals == 1)
                 overlap=overlap0;
             else
                overlap=[overlap overlap0];
             end
             latest_mask(izz0,ixx0)=1;
             fraction_mask(izz0,ixx0)=0;
        end

        % Current non-masked 
        rk_passed=find(latest_mask<1);
        IvalsCorrected =tfmdata(rk_passed,iview);
        %IvalsCorrected=Ivals./attenMaps.data(rk_passed,iview);

        sigma=fn_rayleigh_mle(IvalsCorrected(:)); 

        %termination condition check
        perc_frac=perc_search/100.0;
        threshold2=sigma*sqrt(-2*log(1-perc_frac));
        rk_passed2=find(IvalsCorrected(:)>= threshold2);
        val=100.0*length(rk_passed2)/length(IvalsCorrected(:));
        if (text_output>0);disp(['Iter ',int2str(i),' sigma ',num2str(20*log10(sigma)),' frac ',num2str(val)]); end  

        if (val < 100-perc_search || ii>0)
            if (val-0.05 > 100-perc_search)
                nRemovals2=nRemovals;
                ii=0;
                continue;
            end
            ii=ii+1;

            if (ii>2)
                break;
            end
            latest_mask=latest_mask_old;
            nRemovals2=ceil(nRemovals2*(val_old-(100-perc_search))/(val_old-val));
            nRemovals2=max(1,nRemovals2);
        end     
    end

    latest_mask=1-latest_mask;
    
    % Fill in tiny masked regions/voids
    %options_with_precalcs.hole_fill=10;
    
    if (hole_fill>0)
        n1=length(mesh.z)+2;    % Need to ensure any edge pixels aren't filled in - so add continuous border around actual image
        n2=length(mesh.x)+2;
        tmp=ones(n1,n2);
        tmp(2:n1-1,2:n2-1)=1-reshape(latest_mask,length(mesh.z),length(mesh.x));
        tmp=1-double(bwareaopen(tmp,hole_fill)); %Note: needs to be 'double' not 'int32' otherwise NaN assignment will not work
        latest_mask=tmp(2:n1-1,2:n2-1);
    end

    % Expand masked region by specified amount
    %options_with_precalcs.hole_expand=2;
    if (hole_expand>0)
        ip=hole_expand; %round(options_with_precalcs.hole_expand/options_with_precalcs.pixel_size);
        SE2 = strel('sphere',ip);
        latest_mask = 1-imdilate(reshape(1-latest_mask,length(mesh.z),length(mesh.x)),SE2);
    end
    
    tt=toc(tstart2);
    if (text_output>0);disp(['Time taken for HICS component ',num2str(tt)]); end
    rk_passed=find(latest_mask>0);
    latest_mask(latest_mask<1)=NaN;
    IvalsCorrected =tfmdata(rk_passed,iview).';
    %sigma_o(iview)=fn_rayleigh_mle(Ivals);
    stored_mask(:,iview)=latest_mask(:);
    %IvalsCorrected=tfmdata(rk_passed,iview)./attenMaps.data(rk_passed,iview);
    %[IvalsCorrected,planeParameters(:,iview)]=fn_correct_for_spatial_variation(Ivals,xCoords,yCoords,refCoordTranslation);
    sigma_c(iview)=fn_rayleigh_mle(IvalsCorrected);
end

end