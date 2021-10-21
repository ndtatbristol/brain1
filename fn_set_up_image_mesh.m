function [ax, ms] = fn_set_up_image_mesh(sz_info, create_3d_mesh)
%SUMMARY
%   Sets up image axes (2D or 3D) and meshgrid
%INPUTS
%   sz_info - structure containing fields {x,y,z}_size, {x,y,z}_offset, and
%   pixel_size that are used to define axes
%   create_3d_mesh - 1 to create 3D mesh, 0 for 2D mesh
%OUTPUTS
%   ax - structure with fields x and y (and z) of axis values
%   ms - structure with 2D or 3D fields x and y (and z)
%--------------------------------------------------------------------------

if create_3d_mesh
    ax.x = [-sz_info.x_size / 2: sz_info.pixel_size: sz_info.x_size / 2] + sz_info.x_offset;
    ax.y = [-sz_info.y_size / 2: sz_info.pixel_size: sz_info.y_size / 2] + sz_info.y_offset;
    ax.z = [0: sz_info.pixel_size: sz_info.z_size] + sz_info.z_offset;
    [ms.x, ms.y, ms.z] = meshgrid(ax.x, ax.y, ax.z);
else
    ax.x = [-sz_info.x_size / 2: sz_info.pixel_size: sz_info.x_size / 2] + sz_info.x_offset;
    ax.z = [0: sz_info.pixel_size: sz_info.z_size] + sz_info.z_offset;
    ax.y=0;
    [ms.x, ms.z] = meshgrid(ax.x, ax.z);
end

end