function g = fn_CPU_BP_DAS_sum(exp_data,Focal,Interpolation)

% Parameters

N = length(exp_data.array.el_xc);

Nt = length(exp_data.time);

dt = single(exp_data.time(2) - exp_data.time(1));
t0 = single(exp_data.time(1));

g = 0;
switch fn_determine_exp_data_type(exp_data)
    case 'FMC'
        switch Interpolation
            case 'nearest'
                for n =  1 : length(exp_data.rx) % loop over pairs tx-rx
                    tn = floor((Focal.Law(:,n)-t0)/dt);

                    % verification if tn is in the available range
                    AA = (tn>=0 & tn<Nt-1);
                    tn(~AA)=Nt-1;

                    idx = tn + (exp_data.rx(n)-1)*Nt + (exp_data.tx(n)-1)*N*Nt;
                    Coeff = Focal.Amp(:,n).*AA;

                    g = g + exp_data.time_data(idx+1).*Coeff;

                end
            otherwise % linear interpolation
                for n =  1 : length(exp_data.rx) % loop over pairs tx-rx
                    tn = floor((Focal.Law(:,n)-t0)/dt);

                    % verification if tn is in the available range
                    AA = (tn>=0 & tn<Nt-1);
                    tn(~AA)=Nt-1;

                    idx = tn + (exp_data.rx(n)-1)*Nt + (exp_data.tx(n)-1)*N*Nt;
                    Coeff = Focal.Amp(:,n).*AA;
                    data1 = exp_data.time_data(idx+1);
                    data2 = exp_data.time_data(idx+2);
                    datai = data1 + (data2-data1)/dt.*(Focal.Law(:,n)-(tn*dt+t0));

                    g = g + datai.*Coeff;

                end
        end
    case 'HMC'
        switch Interpolation
            case 'nearest'
                for n =  1 : length(exp_data.rx) % loop over pairs tx-rx
                    
                    idxHMC = (exp_data.tx(n)-1)*N+exp_data.rx(n)-1 - exp_data.tx(n)*(exp_data.tx(n)-1)/2;
                    
                    % Processing of the lower diagonal elements of FMC
                    tn = floor((Focal{1}.Law(:,n)-t0)/dt);
                    AA = (tn>=0 & tn<Nt-1);
                    tn(~AA)=Nt-1;

                    idx = tn + idxHMC*Nt;
                    
                    Coeff = Focal{1}.Amp(:,n).*AA;

                    tmp = exp_data.time_data(idx+1).*Coeff;
                    
                    % Processing of the fictive upper diagonal elements of FMC
                    tn = floor((Focal{2}.Law(:,n)-t0)/dt);

                    AA = (tn>=0 & tn<Nt-1);
                    tn(~AA)=Nt-1;

                    idx = tn + idxHMC*Nt;
                    
                    Coeff = Focal{2}.Amp(:,n).*AA;

                    % Final sum
                    g = g + tmp + exp_data.time_data(idx+1).*Coeff;

                end
            otherwise % linear interpolation
                for n =  1 : length(exp_data.rx) % loop over pairs tx-rx
                    
                    idxHMC = (exp_data.tx(n)-1)*N+exp_data.rx(n)-1 - exp_data.tx(n)*(exp_data.tx(n)-1)/2;
                    
                    tn = floor((Focal{1}.Law(:,n)-t0)/dt);

                    AA = (tn>=0 & tn<Nt-1);
                    tn(~AA)=Nt-1;

                    idx = tn + idxHMC*Nt;
                    Coeff = Focal{1}.Amp(:,n).*AA;
                    data1 = exp_data.time_data(idx+1);
                    data2 = exp_data.time_data(idx+2);
                    datai = data1 + (data2-data1)/dt.*(Focal{1}.Law(:,n)-(tn*dt+t0));
                    tmp = datai.*Coeff;
                    
                    tn = floor((Focal{2}.Law(:,n)-t0)/dt);

                    AA = (tn>=0 & tn<Nt-1);
                    tn(~AA)=Nt-1;

                    idx = tn + idxHMC*Nt;
                    Coeff = Focal{2}.Amp(:,n).*AA;
                    data1 = exp_data.time_data(idx+1);
                    data2 = exp_data.time_data(idx+2);
                    datai = data1 + (data2-data1)/dt.*(Focal{2}.Law(:,n)-(tn*dt+t0));

                    g = g + tmp + datai.*Coeff;

                end
        end
    otherwise
        error('Use FMC or HMC')
end
