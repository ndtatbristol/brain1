function varargout = fn_8contact_BPDAS_wrapper(exp_data, options, mode)
%SUMMARY
%   This wrapper calculates the generalised image based on the
%   back-propagation method using a delay-and-sum approach. Only available
%   for for full or half matrix captures (FMC or HMC).

%USAGE (depending on value of mode argument)
%   initial_info = fn_basic_wrapper([], [], 'return_name_only')
%   extended_info = fn_basic_wrapper(exp_data, [], 'return_info_only')
%   [data, options_with_precalcs] = fn_basic_wrapper(exp_data, options, 'recalc_and_process')
%   data = fn_tfm_wrapper(exp_data, options_with_precalcs, 'process_only')

%UPDATES by PDW 12/11/20 - various alterations to cater for new style
%material velocity descriptions (based on spherical harmonics, even though
%these are not yet used in the generalised imaging calculation and also to
%use the global GPU_PRESENT  variable rather than making local checks.
%Removed input to turn GPU on or off if available as this is now
%automatically added for all imaging functions by giu_process_window 

% default_options.options_changed = 1; %this enables recurring data (e.g. distances to field points to be calculated as a field in options and only recalculated if options_changed = 1)

%the following is the data to allow the processing parameters to be
%displayed and edited in the GUI as well as the default values.
name = 'Contact Generalised image'; %name of process that appears on menu
name_str=['Imaging: ' name];
switch mode
    case 'return_name_only'
        varargout{1} = name;
        return;
        
    case 'return_info_only'
        info = fn_return_info(exp_data);
        info.name = name;
        varargout{1} = info;
        return;
        
    case 'recalc_and_process'
        options_with_precalcs = fn_return_options_with_precalcs(exp_data, options);
        plothan=findall(0,'Name',name_str);
        im_han=findall(plothan,'Type','image');
        options_with_precalcs.ax_han=get(im_han,'Parent');
        
        data = fn_process_using_precalcs(exp_data, options_with_precalcs);
        varargout{1} = data;
        varargout{2} = options_with_precalcs;
        
    case 'process_only'
        data = fn_process_using_precalcs(exp_data, options);
        varargout{1} = data;
        
end
end

%--------------------------------------------------------------------------

function options_with_precalcs = fn_return_options_with_precalcs(exp_data, options)

global GPU_PRESENT

options_with_precalcs = options; %need this line as initial options

%old velocity form
% exp_data.ph_velocity = options_with_precalcs.ph_velocity;
%new velocity form
if isfield(exp_data, 'material')
    options_with_precalcs.original_material = exp_data.material;
elseif isfield(exp_data, 'vel_elipse')
    lebedev_quality = 3;
    options_with_precalcs.original_material.vel_spherical_harmonic_coeffs = fn_spherical_harmonics_for_elliptical_profile(exp_data.vel_elipse(1), exp_data.vel_elipse(1), exp_data.vel_elipse(2), lebedev_quality);
elseif isfield(exp_data, 'ph_velocity')
    options_with_precalcs.original_material.vel_spherical_harmonic_coeffs = exp_data.ph_velocity; %legacy files
else
    error('No valid velocity description found');
end
exp_data.material.vel_spherical_harmonic_coeffs = options_with_precalcs.const_velocity;

%check style of data and array
using_2d_array = any(exp_data.array.el_yc);
data_is_csm = length(unique(exp_data.tx)) == 1;


%set up grid and image axes
if using_2d_array
    error('Not implemented for 2D arrays');
else
    options_with_precalcs.data.x = [-options.x_size / 2: options.pixel_size: options.x_size / 2] + options.x_offset;
    options_with_precalcs.data.z = [0: options.pixel_size: options.z_size] + options.z_offset;
    options_with_precalcs.data.y=0;
    [tmp_mesh.x, tmp_mesh.z] = meshgrid(options_with_precalcs.data.x, options_with_precalcs.data.z);
end

options_with_precalcs.focal_law.interpolation_method = lower(options.interpolation_method);
options_with_precalcs.focal_law.filter_on = options.filter_on;
options_with_precalcs.focal_law.filter = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_half_bandwidth / 2);
if options.angle_limit_on
    options_with_precalcs.focal_law.angle_limit = options.angle_limit;
    options_with_precalcs.focal_law.amp_on = 1;
    dx = max(options_with_precalcs.data.z) * sin(options_with_precalcs.focal_law.angle_limit);
    options_with_precalcs.geom.lines(1).x = [min(exp_data.array.el_xc), max(exp_data.array.el_xc); ...
        min(exp_data.array.el_xc) + dx, max(exp_data.array.el_xc) - dx];
    options_with_precalcs.geom.lines(1).y = zeros(2,2);
    options_with_precalcs.geom.lines(1).z = [0, 0; ones(1,2) * max(options_with_precalcs.data.z)];
    options_with_precalcs.geom.lines(1).style = ':';
end
options_with_precalcs.geom.array = fn_get_array_geom_for_plots(exp_data.array);

%GPU check - need to add disable key as well
if GPU_PRESENT
    if ~isfield(options_with_precalcs.focal_law, 'thread_size')
        options_with_precalcs.focal_law.thread_size=128;
    end
    bit_ver=mexext;
    ptx_file=['gpu_BP_DAS_' bit_ver([end-1:end]) '.ptx'];
   
    switch fn_determine_exp_data_type(exp_data)
        case 'HMC'
            options_with_precalcs.focal_law.kern ...
             = parallel.gpu.CUDAKernel(ptx_file,...
                 'gpu_BP_DAS.cu',...
                 'BP_DAS_FWD_HMC_complex');
        case 'FMC'
            options_with_precalcs.focal_law.kern ...
             = parallel.gpu.CUDAKernel(ptx_file,...
                'gpu_BP_DAS.cu',...
                'BP_DAS_FWD_FMC_complex');
        otherwise
            error('FMC or HMC data expected');
    end

else
    if options_with_precalcs.angle_limit_on
        options_with_precalcs.focal_law.tof = ...
        fn_CPU_BP_DAS_FWD_laws(exp_data,options_with_precalcs.data.x,options_with_precalcs.data.z,options.angle_limit);
    else
        options_with_precalcs.focal_law.tof = ...
        fn_CPU_BP_DAS_FWD_laws(exp_data,options_with_precalcs.data.x,options_with_precalcs.data.z,0);
    end
end

end

%--------------------------------------------------------------------------

function data = fn_process_using_precalcs(exp_data, options_with_precalcs)
%put the actual imaging calculations here, making use of pre-calculated
%values in the options_with_precalcs fields if required.

%copy output output coordinates
data.x = options_with_precalcs.data.x;
if isfield(options_with_precalcs.data, 'y')
    data.y = options_with_precalcs.data.y;
end
data.z = options_with_precalcs.data.z;
%the actual calculation


% Extracting parameters from exp_data
N = length(exp_data.array.el_xc);
d = exp_data.array.el_xc(2) - exp_data.array.el_xc(1);
Nt = length(exp_data.time);
% c = exp_data.ph_velocity;
if isfield(exp_data, 'vel_elipse') %for legacy files, the spherical harmonic coeffs are not defined for ellipse at this point, so need to read default values from legacy info
    c = exp_data.ph_velocity;
elseif (isfield(exp_data, 'material') && isfield(exp_data.material, 'vel_spherical_harmonic_coeffs'))
    [c, ~, ~, ~] = fn_get_nominal_velocity(exp_data.material.vel_spherical_harmonic_coeffs);
elseif isfield(exp_data, 'ph_velocity')
    c = exp_data.ph_velocity;
else
    error('No valid velocity description found');
end
dt = exp_data.time(2) - exp_data.time(1);
t0 = exp_data.time(1);

if isfield(options_with_precalcs, 'use_gpu')
    use_gpu = options_with_precalcs.use_gpu;
else
    use_gpu = 0;
end


% Imaging range parameters
XX = data.x;
ZZ = data.z;
dx = XX(2)-XX(1);
dz = ZZ(2)-ZZ(1);
x0 = XX(1);
z0 = ZZ(1);
Nx = length(XX);
Nz = length(ZZ);

% Definition of filters
omega = 2*pi*(0:Nt-1)'/Nt/dt;
Filter_Gaussian = 2*fn_calc_filter(exp_data.time, options_with_precalcs.centre_freq, options_with_precalcs.centre_freq * options_with_precalcs.frac_half_bandwidth / 2);
Filter_Derivative = 1i*omega/c;
Filter = Filter_Gaussian.*Filter_Derivative;

AngFilter = options_with_precalcs.angle_limit*180/pi;

% Imaging processing on GPU or CPU
if use_gpu
    
    % GPU version based on DAS approach of the BP method
    
    % Threads per block definition
    options_with_precalcs.focal_law.kern.ThreadBlockSize = [4 4 4];
    options_with_precalcs.focal_law.kern.GridSize = ceil([Nz Nx Nx]./options_with_precalcs.focal_law.kern.ThreadBlockSize);
    
    % Prepare input and output
    data_gpu = gpuArray(single(exp_data.time_data));
    g_gpu = gpuArray(complex(zeros(Nz,Nx,Nx,'single')));
    
    % Filter data -- Gaussian (frequency of interest) and derivative (BP-DAS method) filters
    filtered_data_gpu = ifft(fft(data_gpu).*gpuArray(single(Filter))); % complex data
    
    % Evaluate kernel -- use angular filter if AngFilter!=0
    if options_with_precalcs.angle_limit_on
        g = feval(options_with_precalcs.focal_law.kern,g_gpu,filtered_data_gpu,N,Nx,Nz,Nt,d,dx,dz,dt,c,x0,z0,t0,AngFilter);
    else
        g = feval(options_with_precalcs.focal_law.kern,g_gpu,filtered_data_gpu,N,Nx,Nz,Nt,d,dx,dz,dt,c,x0,z0,t0,0);
    end
    g = gather(g);
else
    % CPU version based on DAS approach of the BP method
    
    % Filter data -- Gaussian (frequency of interest) and derivative (BP-DAS method) filters
    filt_exp_data = exp_data;
    filt_exp_data.time_data = single(ifft(fft(exp_data.time_data).*Filter));
    
    % Evaluate the function -- sum over the tx-rx
    g = fn_CPU_BP_DAS_FWD_sum(filt_exp_data,options_with_precalcs.focal_law.tof,options_with_precalcs.focal_law.interpolation_method);
    
    if strcmp(fn_determine_exp_data_type(exp_data),'HMC')
        % if HMC, g contains only non redundant information (xT<=xR)
        % g in xT<=xR is mirrored 
        tmp = zeros(length(ZZ),length(XX),length(XX),'single');
        [~,xR,xT] = ndgrid(ZZ,XX,XX);
        tmp(xR>=xT)=g(:);
        g = tmp;
        g = reshape(g,[length(ZZ),length(XX),length(XX)]);
        for n = 1 : length(XX)
            for m = 1 : n
                g(:,m,n) = g(:,n,m);
            end
        end
    else
        g = reshape(g,[length(ZZ),length(XX),length(XX)]);
    end

end


data.x = XX;
data.y = XX;
data.z = ZZ;

data.f = permute(g,[3 2 1]);

data.geom = options_with_precalcs.geom;
end

%--------------------------------------------------------------------------

function info = fn_return_info(exp_data)

%deal with various velocity forms
if isfield(exp_data, 'vel_elipse') %for legacy files, the spherical harmonic coeffs are not defined for ellipse at this point, so need to read default values from legacy info
    v = exp_data.ph_velocity;
    v_x = exp_data.vel_elipse(1);
    v_y = exp_data.vel_elipse(1);
    v_z = exp_data.vel_elipse(2);
elseif (isfield(exp_data, 'material') && isfield(exp_data.material, 'vel_spherical_harmonic_coeffs'))
    [v, v_x, v_y, v_z] = fn_get_nominal_velocity(exp_data.material.vel_spherical_harmonic_coeffs);
elseif isfield(exp_data, 'ph_velocity')
    v = exp_data.ph_velocity;
    v_x = v;
    v_y = v;
    v_z = v;
else
    error('No valid velocity description found');
end

info.fn_display = @gui_3d_plot_panel;
info.display_options.interpolation = 0;
no_pixels = 30;

info.display_options.axis_equal = 1;
info.display_options.x_axis_sf = 1e3;
info.display_options.y_axis_sf = 1e3;
info.display_options.z_axis_sf = 1e3;
if fn_test_if_gpu_present_and_working
    info.display_options.gpu=1;
end

if isempty(exp_data)
    varargout{1} = [];
    varargout{2} = info;
    return %this is the exit point if exp_data does not exist
end

im_sz_z = max(exp_data.time) * v / 2;
im_sz_xy = max([max(exp_data.array.el_xc) - min(exp_data.array.el_xc), ...
    max(exp_data.array.el_yc) - min(exp_data.array.el_yc)]);
info.options_info.x_size.label = 'X size (mm)';
info.options_info.x_size.default = im_sz_xy;
info.options_info.x_size.type = 'double';
info.options_info.x_size.constraint = [1e-3, 10];
info.options_info.x_size.multiplier = 1e-3;

info.options_info.x_offset.label = 'X offset (mm)';
info.options_info.x_offset.default = 0;
info.options_info.x_offset.type = 'double';
info.options_info.x_offset.constraint = [-10, 10];
info.options_info.x_offset.multiplier = 1e-3;

if any(exp_data.array.el_yc)
    info.options_info.y_size.label = 'Y size (mm)';
    info.options_info.y_size.default = im_sz_xy;
    info.options_info.y_size.type = 'double';
    info.options_info.y_size.constraint = [1e-3, 10];
    info.options_info.y_size.multiplier = 1e-3;
    
    info.options_info.y_offset.label = 'X offset (mm)';
    info.options_info.y_offset.default = 0;
    info.options_info.y_offset.type = 'double';
    info.options_info.y_offset.constraint = [-10, 10];
    info.options_info.y_offset.multiplier = 1e-3;
end

info.options_info.z_size.label = 'Z size (mm)';
info.options_info.z_size.default = im_sz_z;
info.options_info.z_size.type = 'double';
info.options_info.z_size.constraint = [1e-3, 10];
info.options_info.z_size.multiplier = 1e-3;

info.options_info.z_offset.label = 'Z offset (mm)';
info.options_info.z_offset.default = 0;
info.options_info.z_offset.type = 'double';
info.options_info.z_offset.constraint = [-10, 10];
info.options_info.z_offset.multiplier = 1e-3;

info.options_info.filter_on.label = 'Filter';
info.options_info.filter_on.type = 'bool';
info.options_info.filter_on.constraint = {'On', 'Off'};
info.options_info.filter_on.default = 1;

info.options_info.centre_freq.label = 'Filter freq (MHz)';
if isfield(exp_data.array, 'centre_freq')
    info.options_info.centre_freq.default = exp_data.array.centre_freq;
else
    info.options_info.centre_freq.default = 5e6;
end;
info.options_info.centre_freq.type = 'double';
info.options_info.centre_freq.constraint = [0.1, 20e6];
info.options_info.centre_freq.multiplier = 1e6;

info.options_info.frac_half_bandwidth.label = 'Percent b/width';
info.options_info.frac_half_bandwidth.default = 2;
info.options_info.frac_half_bandwidth.type = 'double';
info.options_info.frac_half_bandwidth.constraint = [0.01, 10];
info.options_info.frac_half_bandwidth.multiplier = 0.01;

info.options_info.const_velocity.label = 'Velocity (m/s)';
info.options_info.const_velocity.default = v;
info.options_info.const_velocity.type = 'double';
info.options_info.const_velocity.constraint = [1, 20000];
info.options_info.const_velocity.multiplier = 1;

if no_pixels == 30
    % pixel size equals the wavelength at centre_freq
    info.options_info.pixel_size.default = info.options_info.const_velocity.default/info.options_info.centre_freq.default;
else
    cen_freq=info.options_info.centre_freq.default;
    max_freq = cen_freq + cen_freq .* info.options_info.frac_half_bandwidth.default ./ 2;
    info.options_info.pixel_size.default = (info.options_info.const_velocity.default ./ max_freq) /4; %divide by four is to account for nyquist frequency and out and back path length;
    round_to=1e-5;
    info.options_info.pixel_size.default = floor(info.options_info.pixel_size.default / round_to)*round_to;
end
info.options_info.pixel_size.label = 'Pixel size (mm)';
info.options_info.pixel_size.type = 'double';
info.options_info.pixel_size.constraint = [1e-6, 1];
info.options_info.pixel_size.multiplier = 1e-3;


info.options_info.angle_limit_on.label = 'Angle limiter';
info.options_info.angle_limit_on.type = 'bool';
info.options_info.angle_limit_on.constraint = {'On', 'Off'};
info.options_info.angle_limit_on.default = 0;

info.options_info.angle_limit.label = 'Angle limit (degs)';
info.options_info.angle_limit.default = 30 * pi / 180;
info.options_info.angle_limit.type = 'double';
info.options_info.angle_limit.constraint = [1, 89] * pi / 180;
info.options_info.angle_limit.multiplier = pi / 180;

info.options_info.interpolation_method.label = 'Interpolation';
info.options_info.interpolation_method.default = 'Linear';
info.options_info.interpolation_method.type = 'constrained';
info.options_info.interpolation_method.constraint = {'Linear', 'Nearest'};

%Note: option to switch GPU on/off is automatically added by gui_process_window
%if GPU is present. No need to add it here.


% info.options_info.use_gpu_if_available.label = 'Use GPU if available';
% info.options_info.use_gpu_if_available.type = 'bool';
% info.options_info.use_gpu_if_available.constraint = {'On', 'Off'};
% info.options_info.use_gpu_if_available.default = 1;

end
