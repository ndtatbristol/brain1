function result=fn_tfm_gpu_nomesh(exp_data,mesh,filter, k)
%global blocksize
global SYSTEM_TYPES

if ~isfield(SYSTEM_TYPES,'mat_ver')
    a=ver('matlab');
    SYSTEM_TYPES.mat_ver=str2num(a.Version);
end

%gpu_han=gpuDevice(1);

%define the area of the image field
img_size=size(mesh.x);
%result=zeros(img_size);

%convert data to required gpuarray data
%result=gpuArray(single(result));
n=gpuArray( int32(size(exp_data.time_data,1)));
combs=gpuArray( int32(size(exp_data.time_data,2)));
tx=gpuArray( int32(exp_data.tx));
rx=gpuArray( int32(exp_data.rx));
vel= single(exp_data.ph_velocity);
vel=gpuArray(vel);

filter=gpuArray(single(filter));
time=gpuArray(single(exp_data.time));
time_data=gpuArray(single(exp_data.time_data));

x_min=gpuArray(single(mesh.x(1)));
y_min=gpuArray(single(mesh.y(1)));
z_min=gpuArray(single(mesh.z(1)));
dx=gpuArray(single(exp_data.dx));%z_vals(1,1,2)-z_vals(1,1,1);

el_x=gpuArray(single(exp_data.array.el_xc));
el_y=gpuArray(single(exp_data.array.el_yc));

if SYSTEM_TYPES.mat_ver>=8
    result=gpuArray.zeros(img_size,'single');
    tt_weight = gpuArray.ones(size(tx),'single');
else
    result=parallel.gpu.GPUArray.ones(img_size,'single');
    tt_weight = parallel.gpu.GPUArray.ones(size(tx),'single');
end
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

% if grid_z ==1
%     entry='tfm_2d';
% else
%     entry='tfm_3d';
% end
%
% bit_ver=mexext;
% ptx_file=['nomesh_tfm' bit_ver([end-1:end]) '.ptx'];
%
% k = parallel.gpu.CUDAKernel(ptx_file, 'nomesh_tfm.cu',entry);

k.ThreadBlockSize = 128;%gpu_han.MaxThreadsPerBlock;%TFM_focal_law.thread_size;
k.GridSize = ceil(pixs./k.ThreadBlockSize(1));

real_result=result;
imag_result=result;
[real_result imag_result]=feval(k,real_result,imag_result,n,combs,real_exp,img_exp,tx, rx, vel,time, pixels, grid_x, grid_y, grid_z, x_min, z_min, y_min,dx, el_x, el_y);
result=real_result+imag_result.*1i;
if length(img_size)==3
    result=gather(result);
    result=double(result);
end

end