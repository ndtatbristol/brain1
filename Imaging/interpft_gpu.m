function y = interpft_gpu(x,ny,dim)
global SYSTEM_TYPES

if ~isfield(SYSTEM_TYPES,'mat_ver')
    a=ver('matlab');
    SYSTEM_TYPES.mat_ver=str2num(a.Version);
end

perm = [dim:max(length(size(x)),dim) 1:dim-1];
x = permute(x,perm);

[m,n] = size(x);

%  If necessary, increase ny by an integer multiple to make ny > m.
if ny > m
    a = fft(x,[],1);
    nyqst = ceil((m+1)/2);
    if SYSTEM_TYPES.mat_ver>=8
        b = [a(1:nyqst,:) ; gpuArray.zeros(ny-m,n) ; a(nyqst+1:m,:)];
    else
        b = [a(1:nyqst,:) ; parallel.gpu.GPUArray.zeros(ny-m,n) ; a(nyqst+1:m,:)];
    end
    
    if rem(m,2) == 0
        b(nyqst,:) = b(nyqst,:)/2;
        b(nyqst+ny-m,:) = b(nyqst,:);
    end
    y = ifft(b,[],1);
    if isreal(x), y = real(y); end
    y = y * ny / m;
    
    y = ipermute(y,perm);
else
    y = ipermute(x,perm);
end
end
