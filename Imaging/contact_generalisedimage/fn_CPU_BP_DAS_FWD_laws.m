function Focal = fn_CPU_BP_DAS_FWD_laws(exp_data,XX,ZZ,angFilt)
if isfield(exp_data, 'vel_elipse') %for legacy files, the spherical harmonic coeffs are not defined for ellipse at this point, so need to read default values from legacy info
    c = exp_data.ph_velocity;
elseif (isfield(exp_data, 'material') && isfield(exp_data.material, 'vel_spherical_harmonic_coeffs'))
    [c, ~, ~, ~] = fn_get_nominal_velocity(exp_data.material.vel_spherical_harmonic_coeffs);
elseif isfield(exp_data, 'ph_velocity')
    c = exp_data.ph_velocity;
else
    error('No valid velocity description found');
end

c = single(c);
d = single(exp_data.array.el_xc(2)-exp_data.array.el_xc(1));
Xe = single(exp_data.array.el_xc);

NormCoeff = -d^2/(2*pi*c);

[z,xR,xT] = ndgrid(single(ZZ),single(XX),single(XX));

switch fn_determine_exp_data_type(exp_data)
    case 'FMC'
        z = z(:);
        xR = xR(:);
        xT = xT(:);
        Focal.Law = zeros(length(z),length(exp_data.tx),'single');
        Focal.Amp = Focal.Law;
        for n = 1 : length(exp_data.rx)
            Rtx = sqrt( (Xe(exp_data.tx(n))-xT).^2 + z.^2);
            Rrx = sqrt( (Xe(exp_data.rx(n))-xR).^2 + z.^2);
            Focal.Law(:,n) = (Rtx+Rrx)/c;
            Focal.Amp(:,n) = z.^2./sqrt(Rtx.*Rrx)./(Rtx.*Rrx)*NormCoeff;
            
            if abs(angFilt)>0
                idx = (z./Rtx < cos(angFilt))|(z./Rrx < cos(angFilt));
                Focal.Amp(idx,n)=0;
            end
            
        end
    case 'HMC'
        idx = xR>=xT;
        z = z(idx);
        xT = xT(idx);
        xR = xR(idx);
        Focal{1}.Law = zeros(length(z),length(exp_data.tx),'single');
        Focal{2}.Law = zeros(length(z),length(exp_data.tx),'single');
        Focal{1}.Amp = zeros(length(z),length(exp_data.tx),'single');
        Focal{2}.Amp = zeros(length(z),length(exp_data.tx),'single');
        for n = 1 : length(exp_data.rx)
            % Conventional imaging side
            Rtx = sqrt( (Xe(exp_data.tx(n))-xT).^2 + z.^2);
            Rrx = sqrt( (Xe(exp_data.rx(n))-xR).^2 + z.^2);
            Focal{1}.Law(:,n) = (Rtx+Rrx)/c;
            Focal{1}.Amp(:,n) = z.^2./sqrt(Rtx.*Rrx)./(Rtx.*Rrx)*NormCoeff;
            
            if abs(angFilt)>0
                idx = (z./Rtx < cos(angFilt))|(z./Rrx < cos(angFilt));
                Focal{1}.Amp(idx,n)=0;
            end
            
            % Symmetric imaging side xT->xR and xR->xT
            Rtx = sqrt( (Xe(exp_data.tx(n))-xR).^2 + z.^2);
            Rrx = sqrt( (Xe(exp_data.rx(n))-xT).^2 + z.^2);
            Focal{2}.Law(:,n) = (Rtx+Rrx)/c;
            Focal{2}.Amp(:,n) = z.^2./sqrt(Rtx.*Rrx)./(Rtx.*Rrx);
            
            if abs(angFilt)>0
                idx = (z./Rtx < cos(angFilt))|(z./Rrx < cos(angFilt));
                Focal{2}.Amp(idx,n)=0;
            end
            
        end
    otherwise
        error('Use FMC or HMC');
end


