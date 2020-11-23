function [u,t] = fn_CPU_BP_DAS_INV(exp_data,g,XX,ZZ,Twidth,f0)
%%
% This function recovers the array data according to the generalised image
% g(z,xR,xT). In order to reduce the processing time, the time series are
% calculated directly in the time domain [tc-Twidth/2,tc+Twidth/2], where
% tc is the expected times-of-flight to the target at (xc,zc), and Twidth
% is the time width of interest (e.g., width of the pulse). The variable
% tc is different for every combination of tx-rx. Also, the sampling
% frequency is fixed to fs=4*f0.
%
% Outputs:
% u - is the recovered array data of dimensions [N_time,N_tx*N_rx]
% t - is a matrix containing the time vectors of every pair tx-rx.
%
% For example: plot(t(:,n),u(:,n)) displays the time traces of the n-th
% pair tx-rx.

% Parameters
%determine correct veloicty to use
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
Xe = single(exp_data.array.el_xc);
N = length(exp_data.array.el_xc);
dt = exp_data.time(2) - exp_data.time(1);

z0 = ZZ(1);
dx = XX(2)-XX(1);
dz = ZZ(2)-ZZ(1);
Nz = length(ZZ);
Nx = length(XX);

% Coordinates of the target
xc = mean(XX);
zc = mean(ZZ);

% Downsampling data to fs=4*f0
Fs = 1/dt;
Fmin = 4*f0;
if Fs > Fmin
    dt =  1/Fmin;
end

% Time points of interest
tc = sqrt( (Xe(exp_data.tx)-xc).^2+zc^2)/c +sqrt( (Xe(exp_data.rx)-xc).^2+zc^2)/c;
Np = ceil(Twidth/dt);

t = (1:Np)'*dt;
t = t-mean(t) + tc;




% Prepare input vectors
[xRn,xTn] = ndgrid(0:length(XX)-1,0:length(XX)-1);
[~,Rx,Tx] = ndgrid(1:Np,Xe,Xe);

t = t(:);
Rx = Rx(:);
Tx = Tx(:);

xT = xTn(:)*dx+XX(1);
xR = xRn(:)*dx+XX(1);

NormCoeff = dx^2/(2*pi*c);
u = 0;
for n =  1 : length(xR) % loop over pairs xT-xR
    
    z = sqrt( c^4*t.^4-2*c^2*t.^2.*((xT(n)-Tx).^2+(xR(n)-Rx).^2) ...
        + ((xT(n)-Tx).^2-(xR(n)-Rx).^2).^2 )./(2*c*t);
    
    Rtx = sqrt( (Tx-xT(n)).^2 + z.^2);
    Rrx = sqrt( (Rx-xR(n)).^2 + z.^2);
    
    InvFocalLaw = real(z);
    InvFocalAmp = 1./sqrt(Rtx.*Rrx)*NormCoeff;
    
    zn = floor((InvFocalLaw-z0)/dz);
    
    % verification if tn is in the available range
    AA = (zn>=0 & zn<Nz-2);
    zn(~AA)=Nz-2;
    
    idx = zn + xRn(n)*Nz + xTn(n)*Nx*Nz;
    Coeff = InvFocalAmp.*AA;
    data1 = g(idx+1);
    data2 = g(idx+2);
    datai = data1 + (data2-data1)/dz.*(InvFocalLaw-(zn*dz+z0));
    
    u = u + datai.*Coeff;
    
end

u = reshape(u,[Np,N*N]);
t = reshape(t,[Np,N*N]);