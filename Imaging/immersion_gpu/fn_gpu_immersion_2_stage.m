function [result, surface] = fn_gpu_immersion_2_stage(exp_data, mesh, orig_surface, immersion_options, varargin)
global SYSTEM_TYPES

if ~isfield(SYSTEM_TYPES,'mat_ver')
    a=ver('matlab');
    SYSTEM_TYPES.mat_ver=str2num(a.Version);
end

if nargin < 5
    use_gpu_if_available = 1;
else
    use_gpu_if_available = varargin{1};
end

default_immersion_options.centre_freq = exp_data.array.centre_freq;
default_immersion_options.couplant_velocity = 1480;
default_immersion_options.extrapolate_surface = 0;
default_immersion_options.interp_pts_per_sample_wavelength = 0.5; %set to inf to get benchmark result
default_immersion_options.surface_pts_per_sample_wavelength = 5;

immersion_options = fn_set_default_fields(immersion_options, default_immersion_options);
sample_wavelength = exp_data.ph_velocity / immersion_options.centre_freq;
surface = orig_surface;

%upsample surface and extrapolate if nesc
if immersion_options.extrapolate_surface
    min_x = min([mesh.x(1,:)'; surface.x(:)]);
    max_x = max([mesh.x(1,:)'; surface.x(:)]);
else
    min_x = min(surface.x(:));
    max_x = max(surface.x(:));
end
tmp_x = linspace(min_x, max_x, max([ceil((max_x - min_x) / sample_wavelength * immersion_options.surface_pts_per_sample_wavelength), 2]));
surface.z = interp1(surface.x, surface.z, tmp_x, 'spline', 'extrap');
surface.x = tmp_x;

%interpolate specified surface onto grid x values - needs to be generalised
%for 2D case to identify valid points
tmp_surf_z = interp1(surface.x(:), surface.z(:), mesh.x(1,:), 'linear', 'extrap');

%gpu_han=gpuDevice(1);
%the actual focal law calculations
%start defining the gpu data
s_fine_x=gpuArray(single(surface.x));
s_fine_z=gpuArray(single(surface.z));
s_coars_z=gpuArray(single(tmp_surf_z));
mesh_x=gpuArray(single(mesh.x));
mesh_z=gpuArray(single(mesh.z));
x_arr=gpuArray(single(exp_data.array.el_xc));
z_arr=gpuArray(single(exp_data.array.el_zc));
couple_vel=gpuArray(single(immersion_options.couplant_velocity));
mat_vel=gpuArray(single(exp_data.ph_velocity));
num_els_loc=length(exp_data.array.el_xc);
num_els=gpuArray(int32(num_els_loc));

if SYSTEM_TYPES.mat_ver>=8
    foc_law=gpuArray.zeros(size(mesh.x,1),size(mesh.x,2),num_els,'single');
    foc_amp=gpuArray.zeros(size(mesh.x,1),size(mesh.x,2),num_els,'single');
else
    foc_law=parallel.gpu.GPUArray.zeros(size(mesh.x,1),size(mesh.x,2),num_els,'single');
    foc_amp=parallel.gpu.GPUArray.zeros(size(mesh.x,1),size(mesh.x,2),num_els,'single');
end

%specify total number of pixels and the image dimensions
pixels_loc=prod(size(mesh.x));
grid_x=gpuArray(int32(size(mesh.x,1)));
grid_z=gpuArray(int32(size(mesh.x,2)));
grid_surf=gpuArray(int32(length(surface.x)));

if ~isfield(exp_data, 'vel_poly')
    %focal law without angular dependant velocity
    %create gpu kernel
    bit_ver=mexext;
    ptx_file=['gpu_imm_foc' bit_ver([end-1:end]) '.ptx'];
    
    k = parallel.gpu.CUDAKernel(ptx_file, 'gpu_imm_foc.cu');
    k.ThreadBlockSize = 512;%gpu_han.MaxThreadsPerBlock;
    k.GridSize = ceil((pixels_loc*num_els_loc)./k.ThreadBlockSize(1));
    
    [foc_law foc_amp]=feval(k,foc_law,foc_amp,grid_x,grid_z,grid_surf,s_fine_x,s_fine_z, s_coars_z, mesh_x, mesh_z, x_arr...
        , z_arr, couple_vel, mat_vel, num_els);
else
    %focal law with angular dependant velocity
    k = parallel.gpu.CUDAKernel('gpu_imm_foc_varvel.ptx', 'gpu_imm_foc_varvel.cu');
    k.ThreadBlockSize = 128;%gpu_han.MaxThreadsPerBlock;
    k.GridSize = ceil((pixels_loc*num_els_loc)./k.ThreadBlockSize(1));
    
    if isfield(immersion_options, 'angle_limit');
        ang_lim=gpuArray(single(immersion_options.angle_limit));
    else
        ang_lim=gpuArray(single(270));%270
    end
    poly_order = gpuArray(int32(length(exp_data.vel_poly.p)-1));
    poly = gpuArray(single(exp_data.vel_poly.p));
    mu = gpuArray(single(exp_data.vel_poly.mu));
    
    atten = gpuArray(single(-log(10 ^ (- (immersion_options.atten*1e3)/ 20))));
    %keyboard
    [foc_law foc_amp]=feval(k,foc_law,foc_amp,grid_x,grid_z,grid_surf,s_fine_x,s_fine_z, s_coars_z, mesh_x, mesh_z, x_arr...
        , z_arr, couple_vel, mat_vel, num_els, mu, poly_order, poly, ang_lim, atten);
end

%finally the weightings to account for HMC data
focal_law.tt_weight = ones(1, length(exp_data.tx));
for ii = 1:length(focal_law.tt_weight)
    %check if other combo is present
    tx = exp_data.tx(ii);
    rx = exp_data.rx(ii);
    if tx == rx
        continue;
    end
    if isempty(find(exp_data.tx == rx & exp_data.rx == tx))
        focal_law.tt_weight(ii) = 2;
    end
end
focal_law.filter = fn_calc_filter(exp_data.time, immersion_options.centre_freq, immersion_options.centre_freq * immersion_options.frac_half_bandwidth / 2);

%%% calculate the image without fetching data
%define the area of the image field
img_size=size(mesh.x);
ndims=length(img_size);
%result=zeros(img_size);

%convert data to required gpuarray data
%result=gpuArray(single(result));
if SYSTEM_TYPES.mat_ver>=8
    result=gpuArray.zeros(img_size,'single');
else
    result=parallel.gpu.GPUArray.zeros(img_size,'single');
end
n=gpuArray( int32(size(exp_data.time_data,1)));
combs=gpuArray( int32(size(exp_data.time_data,2)));
tx=gpuArray( int32(exp_data.tx));
rx=gpuArray( int32(exp_data.rx));
tt_weight = gpuArray(single(focal_law.tt_weight));
filter=gpuArray(single(focal_law.filter));
time=gpuArray(single(exp_data.time));
time_data=gpuArray(single(exp_data.time_data));
time_data=ifft((filter*tt_weight.*fft(time_data)))*2;

%split analytic data into real and imaginary parts
real_exp=single(real(time_data));
img_exp=single(imag(time_data));
pixs=prod(img_size);

%specify total number of pixels and the image dimensions
pixels=gpuArray(int32(pixs));
grid_x=gpuArray(int32(size(result,1)));
grid_y=gpuArray(int32(size(result,2)));
grid_z=gpuArray(int32(size(result,3)));

bit_ver=mexext;
ptx_file=['gpu_tfm' bit_ver([end-1:end]) '.ptx'];

k = parallel.gpu.CUDAKernel(ptx_file, 'gpu_tfm.cu', 'tfm_linear_norm');
k.ThreadBlockSize = 128;%gpu_han.MaxThreadsPerBlock;
k.GridSize = ceil(pixs./k.ThreadBlockSize(1));

real_result=result;
imag_result=result;

[real_result imag_result]=feval(k,real_result,imag_result,n,combs,real_exp,img_exp,tx, rx, foc_law,time, pixels, grid_x, grid_y, grid_z, foc_amp);

result=real_result+imag_result.*i;
result=gather(result);
end

