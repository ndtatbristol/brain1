function info = fn_2d_smatrix_reversible_imaging_wrapper(exp_data, data, options)

if isempty(exp_data) & isempty(data)
    info.name = '2D S-matrix (Reversible imaging)';
    return;
else
    info = [];
end;

if ~isfield(options, 'select') || size(options.select, 1) ~= 2 || size(options.select, 2) ~= 2
    warndlg('Select image region first','Warning')
    return;
end

switch fn_determine_exp_data_type(exp_data)
    case 'FMC'
        % go
    case 'HMC'
        % go
    otherwise
        error('Use FMC or HMC')
end

%find max in selected part of image passed to routine - this is the point where S-matrix is
%calculated
i1 = min(find(data.x >= min(options.select(:,1))));
i2 = max(find(data.x <= max(options.select(:,1))));
j1 = min(find(data.z >= min(options.select(:,2))));
j2 = max(find(data.z <= max(options.select(:,2))));
data.x = data.x(i1:i2);
data.z = data.z(j1:j2);
data.f = data.f(j1:j2, i1:i2);
% keyboard;
[dummy, ii] = max(abs(data.f));
[dummy, jj] = max(dummy);
ii = ii(jj);
options.x = data.x;
options.z = data.z;
options.xmax = data.x(jj);
options.zmax = data.z(ii);

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

%defaults
default_aperture_elements = 1;
default_ang_pts = 90;
default_density = 2700;
default_speed_ratio = 2;
default_time_width = 2*sqrt((data.x(1)-data.x(end))^2+(data.z(1)-data.z(end))^2)...
    /c;
default_time_width = floor(default_time_width*100e6)/100e6;
default_centre_freq = exp_data.array.centre_freq;
default_el_width = exp_data.array.el_xc(2)-exp_data.array.el_xc(1);
default_correct_for_propagation_dist = 1;
default_correct_for_el_directivity = 1;
default_correct_for_phase = 1;

%save options options exp_data

% pad_factor = default_pad_factor;
% ang_pts = default_ang_pts;

%figure size
width = 600;
height = 300;
table_pos = [0,1/2,1/3,1/2];
result_pos = [0,0,1/3,1/2];
smatrix_pos = [1/3,0,1/3,1];
graph_pos = [2/3,0,1/3,1];

%create figure
p = get(0, 'ScreenSize');
f = figure('Position',[(p(3) - width) / 2, (p(4) - height) / 2, width, height] ,...
    'MenuBar', 'none', ...
    'NumberTitle', 'off', ...
    'ToolBar', 'None', ...
    'Name', ['Analysis:', '2D S-matrix extraction (Reversible imaging)'] ...
    );

%create graph panels
h_smatrix_panel = uipanel(f, 'Units', 'Normalized', 'Position', smatrix_pos);
h_graph_panel = uipanel(f, 'Units', 'Normalized', 'Position', graph_pos);

%results
h_result = uicontrol('Style', 'text', 'Units', 'Normalized', 'Position', result_pos);

[h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(f, table_pos, 'normalized', @fn_new_params);

content_info.centre_freq.label = 'Centre frequency (MHz)';
content_info.centre_freq.default = default_centre_freq;
content_info.centre_freq.type = 'double';
content_info.centre_freq.constraint = [1, 1e12];
content_info.centre_freq.multiplier = 1e6;

content_info.aperture_els.label = 'Aperture elements';
content_info.aperture_els.default = default_aperture_elements;
content_info.aperture_els.type = 'int';
content_info.aperture_els.constraint = [1, length(exp_data.array.el_xc) - 1];

content_info.time_width.label = 'Time width (us)';
content_info.time_width.default = default_time_width;
content_info.time_width.type = 'double';
content_info.time_width.constraint = [0.01e-6, 10e-6];
content_info.time_width.multiplier = 1e-6;

content_info.el_width.label = 'Element width (mm)';
if isfield(exp_data.array, 'el_x2') %this is not always defined
    content_info.el_width.default = abs(max([exp_data.array.el_x2(1) - exp_data.array.el_xc(1), ...
        exp_data.array.el_x1(1) - exp_data.array.el_xc(1)])) * 2;
else
    content_info.el_width.default = default_el_width;
end
content_info.el_width.type = 'double';
content_info.el_width.constraint = [1e-5, 1];
content_info.el_width.multiplier = 1e-3;

content_info.speed_ratio.label = 'Long / shear speed ratio';
content_info.speed_ratio.default = default_speed_ratio;
content_info.speed_ratio.type = 'double';
content_info.speed_ratio.constraint = [1, 10];
content_info.speed_ratio.multiplier = 1;

content_info.correct_for_el_directivity.label = 'Correct for element directivity';
content_info.correct_for_el_directivity.default = default_correct_for_el_directivity;
content_info.correct_for_el_directivity.type = 'bool';
content_info.correct_for_el_directivity.constraint = {'On', 'Off'};

content_info.correct_for_propagation_dist.label = 'Correct for propagation distance';
content_info.correct_for_propagation_dist.default = default_correct_for_propagation_dist;
content_info.correct_for_propagation_dist.type = 'bool';
content_info.correct_for_propagation_dist.constraint = {'On', 'Off'};

h_fn_set_content(content_info);

h_result = uicontrol('Style', 'text', 'Units', 'Normalized', 'Position', result_pos);

ax_sm = axes('Parent', h_smatrix_panel);
ax_gr = axes('Parent', h_graph_panel);

%trigger the calc
h_data_changed();

    function fn_new_params(params)
        options.time_width = params.time_width;
        options.centre_freq = params.centre_freq;
        options.aperture_els = params.aperture_els;
        options.speed_ratio = params.speed_ratio;
        options.correct_for_propagation_dist = params.correct_for_propagation_dist;
        options.correct_for_el_directivity = params.correct_for_el_directivity;
        options.el_width = params.el_width;
        s = fn_2d_s_matrix_backpropagation_method(exp_data, options);
        
        axes(ax_sm);
        cla;
        pcolor(s.phi * 180 / pi, s.phi * 180 / pi, abs(s.m));
        
        shading flat;
        %         c = caxis;
        %         caxis([0, c(2)]);
        axis equal;
        axis tight;
        axis xy;
        xlabel('{\theta}_1');
        ylabel('{\theta}_2');
        hold on;
        
        axes(ax_gr);
        cla;
        di = abs(diag(s.m));
        plot(s.phi * 180 / pi, di);
        axis tight;
        hold on;
        plot([min(s.phi), max(s.phi)] * 180 / pi, [1, 1] * max(di) / 2, 'r');
        [max_di, i2] = max(di);
        i1 = min(find(di > max_di / 2));
        i3 = max(find(di > max_di / 2));
        plot(s.phi(i1) * [1, 1] * 180 / pi, [0, max(di) / 2], 'r');
        plot(s.phi(i2) * [1, 1] * 180 / pi, [0, max(di)], 'r');
        plot(s.phi(i3) * [1, 1] * 180 / pi, [0, max(di) / 2], 'r');
        failed = 0;
        if (i2 == 1) | (i2 == length(di))
            str = 'FAILED: peak out of range';
            str2 = ' ';
        else
            if i1 > 1 & i3 < length(di)
                str = 'Both HM in range';
                dphi = abs(s.phi(i3) - s.phi(i1)) / 2;
            else
                if i1 == 1;
                    str = 'Lower HM point out of range';
                    dphi = abs(s.phi(i3) - s.phi(i2));
                end
                if i3 == length(di)
                    str = 'Upper HM point out of range';
                    dphi = abs(s.phi(i2) - s.phi(i1));
                end
            end
            
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
            
            wavelength = c / params.centre_freq;
            crack_length = fn_crack_length_from_hwhm(dphi) * wavelength;
            str2 = {...
                sprintf('Angle: %i degrees', round(s.phi(i2) * 180 / pi)), ...
                sprintf('HWHM: %i degrees', round(dphi * 180 / pi)), ...
                sprintf('Length: %.2f mm', crack_length * 1e3)};
            axes(ax_sm);
            plot([min(s.phi), max(s.phi)] * 180 / pi, [min(s.phi), max(s.phi)] * 180 / pi, 'w:');
            plot(s.phi(i2) * 180 / pi, s.phi(i2) * 180 / pi, 'wo');
        end
        
        set(h_result, 'String', ...
            {' ', str, str2{:}}, ...
            'ForegroundColor', 'r', ...
            'HorizontalAlignment', 'Left');
    end
end

function s = fn_2d_s_matrix_backpropagation_method(exp_data, options)
global GPU_PRESENT

% Extracting parameters from exp_data
N = length(exp_data.array.el_xc);
d = exp_data.array.el_xc(2) - exp_data.array.el_xc(1);
Nt = length(exp_data.time);
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

f0 = options.centre_freq;
dt = exp_data.time(2) - exp_data.time(1);
t0 = exp_data.time(1);

% Imaging range parameters
Lambda = c/f0;
dx = d;
dz = Lambda/8;
mesh.x = options.x(1) : dx : options.x(end);
mesh.z = options.z(1) : dz : options.z(end);
x0 = mesh.x(1);
z0 = mesh.z(1);
Nx = length(mesh.x);
Nz = length(mesh.z);

xc = options.xmax;
zc = options.zmax;
%xc = mean(mesh.x);
%zc = mean(mesh.z);

angFilt = 0;
Twidth = options.time_width;

% For simplicity, only FMC is considered
if strcmp(fn_determine_exp_data_type(exp_data),'HMC')
    exp_data = fn_expand_hmc_to_fmc(exp_data);
end

if GPU_PRESENT
    
    bit_ver=mexext;
    ptx_file=['gpu_BP_DAS_' bit_ver([end-1:end]) '.ptx'];
    
    kFWD = parallel.gpu.CUDAKernel(ptx_file,'gpu_BP_DAS.cu','BP_DAS_FWD_FMC_complex');
    kFWD.ThreadBlockSize = [4 4 4];
    kFWD.GridSize = ceil([Nz Nx Nx]./kFWD.ThreadBlockSize);
    
    kINV = parallel.gpu.CUDAKernel(ptx_file,'gpu_BP_DAS.cu','BP_DAS_INV_FMC');
    kINV.ThreadBlockSize = [4 4 4];
    kINV.GridSize = ceil([Nt N N]./kINV.ThreadBlockSize);
    
    % Prepare inputs and outputs
    data_gpu = gpuArray(complex(single(exp_data.time_data)));
    g_gpu = gpuArray(complex(zeros(Nz,Nx,Nx,'single')));
    u_gpu = gpuArray(zeros(Nt,N*N,'single'));
    
    % Evaluate kernels
    g = real(feval(kFWD,g_gpu,data_gpu,N,Nx,Nz,Nt,d,dx,dz,dt,c,x0,z0,t0,0));
    u = feval(kINV,u_gpu,g,N,Nx,Nz,Nt,d,dx,dz,dt,c,x0,z0,t0,Twidth,xc,zc);
    
    u = gather(u);
    t = (0:Nt-1)'*dt+t0;
    
    
else
    Interpolation = 'linear';
    % Limiting Twidth, otherwise too heavy to calculate on CPU
    if Twidth>2e-6
        Twidth = 2e-6;
    end
    
    % Forward imaging
    Focal = fn_CPU_BP_DAS_FWD_laws(exp_data,mesh.x,mesh.z,angFilt);
    g     = fn_CPU_BP_DAS_FWD_sum(exp_data,Focal,Interpolation);
    
    % Inverse imaging
    [u,t] = fn_CPU_BP_DAS_INV(exp_data,g,mesh.x,mesh.z,Twidth,f0);
    dt = t(2) - t(1);
end

% Data spectrum at f0
vals = sum(u .* exp(-j*2*pi*f0*t))*(j*2*pi*f0)^2;%*dt

% Scattering matrix calculation

s.m = zeros(N,N);

d1 = sqrt((exp_data.array.el_xc - xc) .^ 2 + (exp_data.array.el_zc - zc) .^ 2);
d2 = d1(exp_data.tx) + d1(exp_data.rx);

k = 2*pi*f0/c;
vals = vals .* exp(j*k*d2);

if options.correct_for_propagation_dist
    vals = vals .* sqrt(d1(exp_data.tx)) .* sqrt(d1(exp_data.rx));
end

if options.correct_for_el_directivity
    %approximation to half-space directivity function
    theta = atan2(exp_data.array.el_xc - xc, exp_data.array.el_zc - zc);
    vals = vals ./ cos(theta(exp_data.tx)) ./ cos(theta(exp_data.rx));
    %and the element width effect
    a = options.el_width;
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
    lambda = c / options.centre_freq;
    vals = vals ./ sinc(a * sin(theta(exp_data.tx)) / lambda) ./ sinc(a * sin(theta(exp_data.rx)) / lambda);
end

%put into a matrix
q = zeros(length(exp_data.array.el_xc), length(exp_data.array.el_xc));
for ii = 1:length(exp_data.tx)
    q(exp_data.tx(ii), exp_data.rx(ii)) = vals(ii);
end

% Smoothed S-matrix
SmatFilt = ones(options.aperture_els);
XaptFilt = ones(1,options.aperture_els)/options.aperture_els;
Xsub = exp_data.array.el_xc;

q = filter2(SmatFilt,q,'valid');
Xsub = conv(Xsub,XaptFilt,'valid');

s.phi = atan2(xc - Xsub, zc);
s.m = q;

end

function crack_length = fn_crack_length_from_hwhm(hwhm)
data = [
    0.0500    0.9460
    0.1000    0.9193
    0.1500    0.9014
    0.2000    0.8836
    0.2500    0.8657
    0.3000    0.8300
    0.3500    0.7586
    0.4000    0.6515
    0.4500    0.5801
    0.5000    0.5444
    0.5500    0.5176
    0.6000    0.4819
    0.6500    0.4462
    0.7000    0.4284
    0.7500    0.4016
    0.8000    0.3748
    0.8500    0.3391
    0.9000    0.3124
    0.9500    0.3034
    1.0000    0.2856
    1.0500    0.2767
    1.1000    0.2677
    1.1500    0.2588
    1.2000    0.2588
    1.2500    0.2410
    1.3000    0.2320
    1.3500    0.2142
    1.4000    0.2053
    1.4500    0.1963
    1.5000    0.1963
    1.5500    0.1874
    1.6000    0.1874
    1.6500    0.1785
    1.7000    0.1785
    1.7500    0.1696
    1.8000    0.1606
    1.8500    0.1517
    1.9000    0.1517
    1.9500    0.1517
    2.0000    0.1428
    2.0500    0.1428
    2.1000    0.1428
    2.1500    0.1428
    2.2000    0.1339
    2.2500    0.1339
    2.3000    0.1249
    2.3500    0.1249
    2.4000    0.1160
    2.4500    0.1160
    2.5000    0.1160
    2.5500    0.1160
    2.6000    0.1160
    2.6500    0.1071
    2.7000    0.1071
    2.7500    0.1071
    2.8000    0.0982
    2.8500    0.0982
    2.9000    0.0982
    2.9500    0.0982
    3.0000    0.0982
    3.0500    0.0982
    3.1000    0.0982
    3.1500    0.0892
    3.2000    0.0892
    3.2500    0.0892
    3.3000    0.0892
    3.3500    0.0892
    3.4000    0.0803
    3.4500    0.0803
    3.5000    0.0803
    3.5500    0.0803
    3.6000    0.0803
    3.6500    0.0803
    3.7000    0.0803
    3.7500    0.0803
    3.8000    0.0803
    3.8500    0.0714
    3.9000    0.0714
    3.9500    0.0714
    4.0000    0.0714
    4.0500    0.0714
    4.1000    0.0714
    4.1500    0.0714
    4.2000    0.0714
    4.2500    0.0714
    4.3000    0.0714
    4.3500    0.0714
    4.4000    0.0714
    4.4500    0.0714
    4.5000    0.0714
    4.5500    0.0714
    4.6000    0.0714
    4.6500    0.0714
    4.7000    0.0714
    4.7500    0.0714
    4.8000    0.0714
    4.8500    0.0714
    4.9000    0.0714
    4.9500    0.0714
    5.0000    0.0714];

if hwhm < min(data(:,2))
    crack_length = min(data(:,1));
    return;
end
if hwhm > max(data(:,2))
    crack_length = max(data(:,1));
    return;
end
crack_length = data(max(find(data(:,2) > hwhm)), 1);
end