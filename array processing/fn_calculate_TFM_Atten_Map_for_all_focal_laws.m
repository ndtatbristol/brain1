function [tfms] = fn_calculate_TFM_Atten_Map_for_all_focal_laws(exp_data,focal_laws,matAtten)

% Code requires FMC assumption, no HMC allowed
nPixel=size(focal_laws.path,2);

tfms.data=zeros(nPixel,focal_laws.count);
for iview=1:focal_laws.count
    itx=focal_laws.raypaths(1,iview);
    irx=focal_laws.raypaths(2,iview);
    lookup_dist_tx=focal_laws.pathDist(:,:,:,itx);
    tx_atten=focal_laws.matMode(:,itx);
    if (focal_laws.matMode(2,itx) == 0)
        %only 1 leg in this focal law
        tx_legs=1;
    else
        tx_legs=2;
    end
    tx_atten=matAtten(tx_atten(1:tx_legs));
    if (itx ~= irx)
        lookup_dist_rx = focal_laws.pathDist(:,:,:,irx);
        if (focal_laws.matMode(2,irx) == 0)
            %only 1 leg in this focal law
            rx_legs=1;
        else
            rx_legs=2;
        end
        rx_atten=focal_laws.matMode(:,irx); 
        rx_atten=matAtten(rx_atten(1:rx_legs));
    else
        rx_atten=0;
    end
    nscanlines=length(exp_data.rx);
    %tx_atten
    use_mex=1;
    if (use_mex>0)
        if (itx == irx)
            %size(mex_atten_map(single(tx_atten),single(tx_atten),int32(exp_data.tx-1),int32(exp_data.rx-1),single(lookup_dist_tx)))
            tfms.data(:,iview)=mex_atten_map(single(tx_atten),single(tx_atten),int32(exp_data.tx-1),int32(exp_data.rx-1),single(lookup_dist_tx));
        else
            tfms.data(:,iview)=mex_atten_map(single(tx_atten),single(rx_atten),int32(exp_data.tx-1),int32(exp_data.rx-1),single(lookup_dist_tx),single(lookup_dist_rx));
        end
        %return
    else
    
        if (itx == irx)
            for iPixel=1:nPixel
                for iScanline=1:nscanlines
                    itx=exp_data.tx(iScanline);
                    irx=exp_data.rx(iScanline);
                    txFactor=1.0;
                    for tx1=1:tx_legs
                        txFactor=txFactor*exp(-tx_atten(tx1)*lookup_dist_tx(tx1,itx,iPixel));
                    end
                    rxFactor=1.0;
                    for rx1=1:tx_legs
                        rxFactor=rxFactor*exp(-tx_atten(rx1)*lookup_dist_tx(rx1,irx,iPixel));
                    end
                    tfms.data(iPixel,iview)=tfms.data(iPixel,iview)+rxFactor*txFactor;
                end
            end
        else
            for iPixel=1:nPixel
                for iScanline=1:nscanlines
                    itx=exp_data.tx(iScanline);
                    irx=exp_data.rx(iScanline);
                    txFactor=1.0;
                    for tx1=1:tx_legs
                        txFactor=txFactor*exp(-tx_atten(tx1)*lookup_dist_tx(tx1,itx,iPixel));
                    end
                    rxFactor=1.0;
                    for rx1=1:rx_legs
                        rxFactor=rxFactor*exp(-rx_atten(rx1)*lookup_dist_rx(rx1,irx,iPixel));
                    end
                    tfms.data(iPixel,iview)=tfms.data(iPixel,iview)+rxFactor*txFactor;
                end
            end
        end
        
        
    end
end

if (use_mex < 1)
    tfms.data=tfms.data/nscanlines;
end



