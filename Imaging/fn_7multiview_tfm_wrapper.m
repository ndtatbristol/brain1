function varargout = fn_7multiview_tfm_wrapper(exp_data, options, mode)
%SUMMARY
%   This picks the basic immersion multiview TFM depending on type of data with
%   minimal extra arguments

%USAGE (depending on value of mode argument)
%   initial_info = fn_basic_wrapper([], [], 'return_name_only')
%   extended_info = fn_basic_wrapper(exp_data, [], 'return_info_only')
%   [data, options_with_precalcs] = fn_basic_wrapper(exp_data, options, 'recalc_and_process')
%   data = fn_tfm_wrapper(exp_data, options_with_precalcs, 'process_only')

% default_options.options_changed = 1; %this enables recurring data (e.g. distances to field points to be calculated as a field in options and only recalculated if options_changed = 1)

%the following is the data to allow the processing parameters to be
%displayed and edited in the GUI as well as the default values.
name = 'Multiview TFM (Immersion)'; %name of process that appears on menu
%force recalc of focal law if in surface measuring mode
update_focal_law_only_if_changed=0;
if strcmp(mode, 'process_only') && isfield(options, 'surface_type') && strcmp(options.surface_type, '|M|easured')% && isfield(options, 'show_couplant_only') && ~options.show_couplant_only
    mode = 'recalc_and_process'; 
    update_focal_law_only_if_changed=1;
end

switch mode
    case 'update_from_file'
    % Process from file, then switch to recalc_and_process
    disp('Updating mask and noise from file')
    if isfield(options,'mask_from_file') && isfield(options,'focal_laws')

        %check bounds are acceptable and all current views are included in mask in file
        if (options.mask_from_file.views_start <= options.focal_laws.views_start && options.mask_from_file.views_end >= options.focal_laws.views_end && options.mask_from_file.x(1) <= options.data.x(1) && options.mask_from_file.x(end) >= options.data.x(end) && options.mask_from_file.z(1) <= options.data.z(1) && options.mask_from_file.z(end) >= options.data.z(end))
            view_offset=options.focal_laws.views_start-options.mask_from_file.views_start;
            options.rms=options.mask_from_file.rms(:,1+view_offset:options.focal_laws.views_end-options.focal_laws.views_start+view_offset+1);
            options.b_fit=options.mask_from_file.b_fit(:,1+view_offset:options.focal_laws.views_end-options.focal_laws.views_start+view_offset+1);
            
            options.attenMaps.data=options.mask_from_file.attenMaps(:,1+view_offset:options.focal_laws.views_end-options.focal_laws.views_start+view_offset+1);
            
            if (abs(options.mask_from_file.x(2)-options.mask_from_file.x(1) - options.pixel_size)<1e-8)
                %same pixel size, no need for interpolation
                x_offset=round((options.data.x(1)-options.mask_from_file.x(1))/options.pixel_size);
                z_offset=round((options.data.z(1)-options.mask_from_file.z(1))/options.pixel_size);
                nviews_file=size(options.mask_from_file.mask,2);
                
                tmp=reshape(options.mask_from_file.mask,length(options.mask_from_file.z),length(options.mask_from_file.x),nviews_file);
                options.mask=tmp(1+z_offset:length(options.data.z)+z_offset,1+x_offset:length(options.data.x)+x_offset,1+view_offset:options.focal_laws.views_end-options.focal_laws.views_start+view_offset+1);
                options.mask=reshape(options.mask,length(options.data.z)*length(options.data.x),options.focal_laws.views_end-options.focal_laws.views_start+1);
            else
                %different pixel sizing, need to interpolate
            end
        
        end
        
    else
        %do nothing about missing information, just process as before
    end
    mode='recalc_and_process';
    update_focal_law_only_if_changed=1;
end

switch mode
    case 'return_name_only'
        varargout{1} = name;
        return;
        
    case 'return_info_only'
        info = fn_return_info(exp_data);
        info.name = name;
        %disp('Adding custom buttons in wrapper')
        %custom buttons
        ii=1;
        info.display_options.custom_button(ii).string='ViewName';
        info.display_options.custom_button(ii).style='togglebutton';
        info.display_options.custom_button(ii).function=@fn_viewnames_active;
        info.display_options.custom_button(ii).defaultValue=1; %selected
        info.display_options.custom_button(ii).enable_function=[];
        ii=ii+1;
        info.display_options.custom_button(ii).string='Masked';
        info.display_options.custom_button(ii).style='togglebutton';
        info.display_options.custom_button(ii).function=@fn_mask_active;
        info.display_options.custom_button(ii).defaultValue=1; %selected
        info.display_options.custom_button(ii).enable_function=@fn_button_mask_enabled;
        ii=ii+1;
        info.display_options.custom_button(ii).string='TFM';
        info.display_options.custom_button(ii).style='radiobutton';
        info.display_options.custom_button(ii).group=1;
        info.display_options.custom_button(ii).function=@fn_tfm_active;
        %info.display_options.custom_button(ii).defaultValue=1; %selected
        info.display_options.custom_button(ii).enable_function=@fn_button_mask_enabled;
        ii=ii+1;
        info.display_options.custom_button(ii).string='Levelled';
        info.display_options.custom_button(ii).style='radiobutton';
        info.display_options.custom_button(ii).group=1;
        info.display_options.custom_button(ii).function=@fn_levelling_active;
        %info.display_options.custom_button(ii).defaultValue=1; %selected
        info.display_options.custom_button(ii).enable_function=@fn_button_mask_enabled;
        ii=ii+1;
        info.display_options.custom_button(ii).string='P-value';
        info.display_options.custom_button(ii).style='radiobutton';
        info.display_options.custom_button(ii).group=1;
        info.display_options.custom_button(ii).function=@fn_pvalues_active;
        %info.display_options.custom_button(ii).defaultValue=0; %selected
        info.display_options.custom_button(ii).enable_function=@fn_button_mask_enabled;
        ii=ii+1;
        info.display_options.custom_button(ii).string='P_OR';
        info.display_options.custom_button(ii).style='togglebutton';
        info.display_options.custom_button(ii).function=@fn_pvalues_combined_active;
        info.display_options.custom_button(ii).defaultValue=1; %selected
        info.display_options.custom_button(ii).enable_function=@fn_pvalues_combined_enabled;
        
        varargout{1} = info;
        return;
        
    case 'recalc_and_process'
        options_with_precalcs = fn_return_options_with_precalcs(exp_data, options,update_focal_law_only_if_changed);
        if isempty(options_with_precalcs)
            data = []; %catch for 2D array which are not supported
        else
            [data,options_with_precalcs] = fn_process_using_precalcs(exp_data, options_with_precalcs);
        end
        varargout{1} = data;
        varargout{2} = options_with_precalcs;

    case 'process_only'
        data = fn_process_using_precalcs(exp_data, options);
        varargout{1} = data;

    
    
    
end
end

%--------------------------------------------------------------------------

function options_with_precalcs = fn_return_options_with_precalcs(exp_data, options,update_focal_law_only_if_changed)
options_with_precalcs = options; %need this line as initial options are copied into options_with_precalcs


exp_data.ph_velocity = options_with_precalcs.ph_velocity;

%set up grid and image axes
if any(exp_data.array.el_yc)
    options_with_precalcs = []; 
    warndlg('2D arrays not yet supported','Warning')
    return;%2D arrays not supported yet
%     options_with_precalcs.data.x = [-options.x_size / 2: options.pixel_size: options.x_size / 2] + options.x_offset;
%     options_with_precalcs.data.y = [-options.y_size / 2: options.pixel_size: options.y_size / 2] + options.y_offset;
%     options_with_precalcs.data.z = [0: options.pixel_size: options.z_size] + options.z_offset;
%     [tmp_mesh.x, tmp_mesh.y, tmp_mesh.z] = meshgrid(options_with_precalcs.data.x, options_with_precalcs.data.y, options_with_precalcs.data.z);
else
    options_with_precalcs.data.x = [0: options.pixel_size: options.x_size] + options.x_offset;
    options_with_precalcs.data.z = [0: options.pixel_size: options.z_size] + options.z_offset;
    [tmp_mesh.x, tmp_mesh.z] = meshgrid(options_with_precalcs.data.x, options_with_precalcs.data.z);
    % form mesh for normalisation region (under array, backwall)
    x1 = [min(exp_data.array.el_xc): options.pixel_size: max(exp_data.array.el_xc)];
    z1 = [options_with_precalcs.backwall_depth-5*options.pixel_size : options.pixel_size : options_with_precalcs.backwall_depth+5*options.pixel_size];
    [tmp_mesh_nf.x, tmp_mesh_nf.z] = meshgrid(x1, z1);
end

data_is_csm = length(unique(exp_data.tx)) == 1;

%generate surface
tmp = exp_data.ph_velocity;
exp_data.ph_velocity = options_with_precalcs.couplant_velocity;
if (options_with_precalcs.filter_on)
    filter_opt=1;
else
    filter_opt=4;
end

try
    options_with_precalcs.original_array;
    if (length(options_with_precalcs.original_array.el_xc) ~= length(exp_data.array.el_xc) || abs(options_with_precalcs.original_array.el_x1(1)- exp_data.array.el_x1(1))>0.01e-3 || abs(options_with_precalcs.original_array.el_x1(end)- exp_data.array.el_x1(end))>0.01e-3)
        options_with_precalcs.original_array=exp_data.array;
        options_with_precalcs.original_array.el_xc=options_with_precalcs.original_array.el_xc-options_with_precalcs.original_array.el_xc(1);
    end
catch
    options_with_precalcs.original_array=exp_data.array;
    options_with_precalcs.original_array.el_xc=options_with_precalcs.original_array.el_xc-options_with_precalcs.original_array.el_xc(1);
end
exp_data.array=options_with_precalcs.original_array;

try
    options_with_precalcs.view_start_old;
catch
    options_with_precalcs.view_start_old=0;
    options_with_precalcs.view_end_old=0;
end

if (~isfield(options_with_precalcs,'probe'))
    update_focal_law_only_if_changed=0;
end
switch options.surface_type
    case '|M|easured'
        [~,probe]=fn_extract_frontwall_signal_and_location(exp_data,options_with_precalcs.instrument_delay*1e-9,options_with_precalcs.min_t*1e-6,10e6,filter_opt,options_with_precalcs.centre_freq * options_with_precalcs.frac_half_bandwidth / 2,options_with_precalcs.centre_freq);
    case '|S|pecified'
        probe.standoff=options_with_precalcs.measured_probe_standoff;
        probe.angle1=options_with_precalcs.measured_probe_angle1;
    otherwise
        probe.standoff=options_with_precalcs.measured_probe_standoff;
        probe.angle1=options_with_precalcs.measured_probe_angle1;
end

exp_data.ph_velocity = tmp;
if (update_focal_law_only_if_changed>0)
    if (abs(probe.standoff-options_with_precalcs.probe.standoff)>0.1e-3 || abs(probe.angle1-options_with_precalcs.probe.angle1)>0.25)
        disp('Probe location changed')
        recalc_focal_law=1;
        options_with_precalcs.measured_probe_standoff=probe.standoff;
        options_with_precalcs.measured_probe_angle1=probe.angle1;
        exp_data.array.el_zc= -probe.standoff-sin(probe.angle1*pi/180)*options_with_precalcs.original_array.el_xc;
        exp_data.array.el_xc = 0.0+cos(probe.angle1*pi/180)*options_with_precalcs.original_array.el_xc;
    else
        recalc_focal_law=0;
    end
else
    recalc_focal_law=1;
    options_with_precalcs.measured_probe_standoff=probe.standoff;
    options_with_precalcs.measured_probe_angle1=probe.angle1;
    exp_data.array.el_zc= -probe.standoff-sin(probe.angle1*pi/180)*options_with_precalcs.original_array.el_xc;
    exp_data.array.el_xc = 0.0+cos(probe.angle1*pi/180)*options_with_precalcs.original_array.el_xc;
end
options_with_precalcs.probe=probe;

%GPU check
if (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0) && (options_with_precalcs.use_gpu_if_available)
    if (~isfield(options_with_precalcs,'gpu'))
        options_with_precalcs.gpu=gpuDevice;
    end
    if isfield(options_with_precalcs, 'sample_focal_law') && ~isfield(options_with_precalcs.sample_focal_law, 'thread_size')
        gpu_han=gpuDevice;
        options_with_precalcs.sample_focal_law.thread_size=gpu_han.MaxThreadsPerBlock;
        
    end
end
if (exist('gpuDeviceCount') == 2) && (gpuDeviceCount > 0) && (options_with_precalcs.use_gpu_if_available)
    if isfield(options_with_precalcs, 'couplant_focal_law') && ~isfield(options_with_precalcs.couplant_focal_law, 'thread_size')
        gpu_han=gpuDevice;
        options_with_precalcs.couplant_focal_law.thread_size=gpu_han.MaxThreadsPerBlock;
    end
end

if data_is_csm
    warndlg('CSM data not yet supported','Warning')
    options_with_precalcs = []; return;%CSM immersion data not supported yet
%need to think how to do this case!
else
    % Create multiview options
    if (options_with_precalcs.view_start_old ~= options_with_precalcs.view_start || options_with_precalcs.view_end_old ~= options_with_precalcs.view_end)
        options_with_precalcs.view_start_old=options_with_precalcs.view_start;
        options_with_precalcs.view_end_old=options_with_precalcs.view_end;
        recalc_focal_law=1;
    end
    if (recalc_focal_law>0)
        
        nview1=options_with_precalcs.view_start;
        nview2=options_with_precalcs.view_end;
        dimensions_opt=2; direction_opt=0; travels_calculated=-1; sample_freq=25e6;
        front_zrange=[0 0];
        front_xrange(1)=min(min(tmp_mesh.x(:)),min(exp_data.array.el_xc)-1e-3);
        front_xrange(2)=max(max(tmp_mesh.x(:)),max(exp_data.array.el_xc)+1e-3);
        wavelength=options_with_precalcs.ph_velocity/options_with_precalcs.centre_freq;
        frontwall.nfrontwall=round((front_xrange(2)-front_xrange(1))/wavelength)*options_with_precalcs.surface_pts_per_sample_wavelength;
        frontwall.nfrontwall=max(frontwall.nfrontwall,30);
        orig_surface.x= linspace(front_xrange(1), front_xrange(2), frontwall.nfrontwall);
        orig_surface.z= linspace(front_zrange(1), front_zrange(2), frontwall.nfrontwall);
        if (nview2>3)
            front_zrange=[options_with_precalcs.backwall_depth options_with_precalcs.backwall_depth];
            backwall.coordX=orig_surface.x;
            backwall.coordZ= linspace(front_zrange(1), front_zrange(2), frontwall.nfrontwall);
        else
            backwall.coordX=0;
            backwall.coordZ=0;
        end
        
        combine_opt=0;
        if (isfield(options_with_precalcs,'gpu'))
            totMem=options_with_precalcs.gpu.TotalMemory;
            npixels=length(tmp_mesh.x(:));
            nelements=length(exp_data.array.el_zc);
            focalMem=4*npixels*nelements*2; %Assume RX ~= TX for this calculation as conservative estimate of size required and single (x4) rather than double (x8) to convert to bytes
            fmcMem=4*2*length(exp_data.time_data(:)); % Assume single precision, complex data
            memRequired=fmcMem+focalMem* (nview2-nview1+1); 
            if (memRequired > 0.8*totMem) % Allow for other data / usage of the GPU Memory
                % don't use combined
                combine_opt=0;
                disp('Cannot use combined focal law')
            else
                combine_opt=1;
                disp('Using combined focal law')
            end
            %combine_opt=0;
        end
            
        if (strcmp(options_with_precalcs.noise_levelling_status,'Off') == 0)
            [options_with_precalcs.focal_laws]=fn_determine_focal_laws_with_dist(nview1,nview2,options_with_precalcs.instrument_delay*1e-9,0,dimensions_opt,direction_opt,exp_data,options_with_precalcs.couplant_velocity,options_with_precalcs.ph_velocity, options_with_precalcs.ph_velocity2,exp_data.array.el_xc,exp_data.array.el_yc,exp_data.array.el_zc,tmp_mesh.x(:),0,tmp_mesh.z(:),orig_surface.x,0,orig_surface.z,backwall.coordX,0,backwall.coordZ,sample_freq,travels_calculated,'single');
        
            nviews=options_with_precalcs.focal_laws.count;
            %disp('Calculating Attenuation Map');
            %disp('Using mean coordinates for plane adjustment')
            matAtten(1)=options_with_precalcs.attenuation_L;
            matAtten(2)=options_with_precalcs.attenuation_T;
            [options_with_precalcs.attenMaps] = fn_calculate_TFM_Atten_Map_for_all_focal_laws(exp_data,options_with_precalcs.focal_laws,matAtten);
            %options_with_precalcs.do_not_update=1;
            nx=length(options_with_precalcs.data.x);
            nz=length(options_with_precalcs.data.z);
            refCoordTranslation(1)=mean(options_with_precalcs.data.x);
            refCoordTranslation(2)=mean(options_with_precalcs.data.z);
            [xVal,xLoc]=min(abs(options_with_precalcs.data.x-refCoordTranslation(1)));
            [zVal,zLoc]=min(abs(options_with_precalcs.data.z-refCoordTranslation(2)));
            ind1=sub2ind([nz,nx],zLoc,xLoc);
            atten_map_norm_factor_roi=options_with_precalcs.attenMaps.data(ind1,:);
            for iview=1:nviews
                options_with_precalcs.attenMaps.data(:,iview)=double(options_with_precalcs.attenMaps.data(:,iview)./atten_map_norm_factor_roi(iview));
            end
        else
            [options_with_precalcs.focal_laws]=fn_determine_focal_laws(0,nview1,nview2,options_with_precalcs.instrument_delay*1e-9,0,dimensions_opt,direction_opt,exp_data,options_with_precalcs.couplant_velocity,options_with_precalcs.ph_velocity, options_with_precalcs.ph_velocity2,exp_data.array.el_xc,exp_data.array.el_yc,exp_data.array.el_zc,tmp_mesh.x(:),0,tmp_mesh.z(:),orig_surface.x,0,orig_surface.z,backwall.coordX,0,backwall.coordZ,sample_freq,travels_calculated,'single');
            nviews=options_with_precalcs.focal_laws.count;
        end  
        % Focal laws associated with normalisation region - region around L-L backwall (or backwall in contact case)
        if (nview1>0)
            nview11=1; nview21=1; % L-L view only
        else
            nview11=0; nview21=0; % Contact view only
        end
        [options_with_precalcs.focal_laws_norm_region]=fn_determine_focal_laws(0,nview11,nview21,options_with_precalcs.instrument_delay*1e-9,0,dimensions_opt,direction_opt,exp_data,options_with_precalcs.couplant_velocity,options_with_precalcs.ph_velocity, options_with_precalcs.ph_velocity2,exp_data.array.el_xc,exp_data.array.el_yc,exp_data.array.el_zc,tmp_mesh_nf.x(:),0,tmp_mesh_nf.z(:),orig_surface.x,0,orig_surface.z,backwall.coordX,0,backwall.coordZ,sample_freq,travels_calculated,'single');
        
        if (combine_opt)
            npixels=length(tmp_mesh.x(:));
            nelements=length(exp_data.array.el_zc);
            %% Combine all focal laws into 1 big focal law
            focal_laws_old=options_with_precalcs.focal_laws;
            options_with_precalcs.focal_laws=[];
            options_with_precalcs.focal_laws.count=1;
            options_with_precalcs.focal_laws.combined=1;
            options_with_precalcs.focal_laws.name=focal_laws_old.name;
            options_with_precalcs.focal_laws.views=nviews;
            options_with_precalcs.focal_laws.pixels=npixels;
            nviews2=nview21-nview11+1;
            options_with_precalcs.focal_laws.views_norm=nviews2;
            npixels2=length(tmp_mesh_nf.z(:));
            options_with_precalcs.focal_laws.pixels_norm=npixels2;
            
            options_with_precalcs.focal_laws.raypaths=[1;2];
            options_with_precalcs.focal_laws.views_start=nview1;
            options_with_precalcs.focal_laws.views_end=nview2;
            %switch (precision)
            %    case 'single'
                    options_with_precalcs.focal_laws.path_tx=single(zeros(nelements,npixels*nviews+nviews2*npixels2));
                    options_with_precalcs.focal_laws.path_rx=single(zeros(nelements,npixels*nviews+nviews2*npixels2));
            %    otherwise
            %        focal_laws.path_tx=zeros(nelements,npixels*nviews);
            %        focal_laws.path_rx=zeros(nelements,npixels*nviews);
            %end
            for i=1:nviews
                tx_path=focal_laws_old.raypaths(1,i); rx_path=focal_laws_old.raypaths(2,i);
                options_with_precalcs.focal_laws.path_tx(:,(i-1)*npixels+1:i*npixels)=focal_laws_old.path(:,:,tx_path);
                options_with_precalcs.focal_laws.path_rx(:,(i-1)*npixels+1:i*npixels)=focal_laws_old.path(:,:,rx_path);
            end
            for i=1:options_with_precalcs.focal_laws.views_norm
                tx_path=options_with_precalcs.focal_laws_norm_region.raypaths(1,i); 
                rx_path=options_with_precalcs.focal_laws_norm_region.raypaths(2,i);
                options_with_precalcs.focal_laws.path_tx(:,npixels*nviews+(i-1)*npixels2+1:npixels*nviews+i*npixels2)=options_with_precalcs.focal_laws_norm_region.path(:,:,tx_path);
                options_with_precalcs.focal_laws.path_rx(:,npixels*nviews+(i-1)*npixels2+1:npixels*nviews+i*npixels2)=options_with_precalcs.focal_laws_norm_region.path(:,:,rx_path);
            end
            options_with_precalcs.focal_laws.path_tx=options_with_precalcs.focal_laws.path_tx.';
            options_with_precalcs.focal_laws.path_rx=options_with_precalcs.focal_laws.path_rx.';
        end
        
        
        options_with_precalcs.filter = fn_calc_filter(exp_data.time, options.centre_freq, options.centre_freq * options.frac_half_bandwidth / 2,40,1);  
    end        

end

nviews=length(options_with_precalcs.focal_laws.name);

image_aspect_ratio=options_with_precalcs.x_size/options_with_precalcs.z_size;
gap=2; %mheight=7;
if (image_aspect_ratio<1)
    c=0; nwidth=0;
    while (c<1)
        nwidth=nwidth+1;
        if (image_aspect_ratio*nwidth>1)
            if (nwidth>1); nwidth=nwidth-1; end;
            break;
        end
    end
    if (nwidth>nviews)
        nwidth=nviews;
        mheight=1;
    else
        sf=sqrt(nviews/nwidth);
        nwidth=ceil(sf*nwidth);
        mheight=ceil(nviews/nwidth);
    end
else
    c=0; mheight=0;
    while (c<1)
        mheight=mheight+1;
        if (image_aspect_ratio*mheight>1)
            if (mheight>1); mheight=mheight-1; end;
            break;
        end
    end
    if (mheight>nviews)
        mheight=nviews;
        nwidth=1;
    else
        sf=sqrt(nviews/mheight);
        mheight=ceil(sf*mheight);
        nwidth=ceil(nviews/mheight);
    end
end    

iwidth=0; iheight=1;
for j=1:nviews
    iwidth=iwidth+1;
    if (iwidth>nwidth)
        iwidth=1; iheight=iheight+1;
    end
end

options_with_precalcs.n_plots=nwidth;
options_with_precalcs.m_plots=iheight;
options_with_precalcs.gap=gap;

%show surface on results
if (recalc_focal_law>0)
    nz=length(options_with_precalcs.data.z);
    front_zrange=options_with_precalcs.backwall_depth;
    dz=options_with_precalcs.data.z(end)-options_with_precalcs.data.z(1);
    if (isfield(options_with_precalcs,'geom'))
        if (isfield(options_with_precalcs.geom,'lines'))
            options_with_precalcs.geom=rmfield(options_with_precalcs.geom,'lines');
        end
    end
    for i=1:iheight %options_with_precalcs.sample_focal_law.views
        %frontwall
        options_with_precalcs.geom.lines((i-1)*2+1).x = orig_surface.x;
        options_with_precalcs.geom.lines((i-1)*2+1).y = zeros(size(orig_surface.x));
        options_with_precalcs.geom.lines((i-1)*2+1).z = orig_surface.z+dz*(i-1)*(nz+gap)/((nz+gap)*(iheight-1)+nz);
        options_with_precalcs.geom.lines((i-1)*2+1).style = '-';
        options_with_precalcs.geom.lines((i-1)*2+1).color = 'r';
        %backwall
        options_with_precalcs.geom.lines(i*2).x = orig_surface.x;
        options_with_precalcs.geom.lines(i*2).y = zeros(size(orig_surface.x));
        options_with_precalcs.geom.lines(i*2).z = orig_surface.z+dz*(i-1)*(nz+gap)/((nz+gap)*(iheight-1)+nz)+(front_zrange/dz)*dz*nz/((nz+gap)*(iheight-1)+nz);
        options_with_precalcs.geom.lines(i*2).style = '-';
        options_with_precalcs.geom.lines(i*2).color = 'g';
    end
    options_with_precalcs.geom.backwall_z=options_with_precalcs.backwall_depth;
    
end

options_with_precalcs.geom.array = fn_get_array_geom_for_plots(exp_data.array);



end

%--------------------------------------------------------------------------

function [data,options_with_precalcs] = fn_process_using_precalcs(exp_data, options_with_precalcs)
%put the actual imaging calculations here, making use of pre-calculated
%values in the options_with_precalcs fields if required.
tstart=tic;
%copy output coordinates
data.x = options_with_precalcs.data.x;
if isfield(options_with_precalcs.data, 'y')
    data.y = options_with_precalcs.data.y;
end
data.z = options_with_precalcs.data.z;

nviews=length(options_with_precalcs.focal_laws.name);

if (nviews>1)
    data.combined_plot=1;
    data.views_start=options_with_precalcs.focal_laws.views_start;
    data.views_end=options_with_precalcs.focal_laws.views_end;
end

data.view_names=options_with_precalcs.focal_laws.name;

if (options_with_precalcs.filter_on == 1)
    filtered_data=ifft(spdiags(options_with_precalcs.filter, 0, length(exp_data.time), length(exp_data.time)) * fft(exp_data.time_data));
else
    filtered_data=hilbert(exp_data.time_data);
end    

% generate sample result 

switch options_with_precalcs.interpolation_method
case 'Nearest'
    int_opt=0;
case 'Linear'
    int_opt=1;
case 'Lanczos (a=2)'
    int_opt=2;
otherwise
    int_opt=3;
end

% Calculate TFMs
tstart2=tic;
if (isfield(options_with_precalcs.focal_laws,'combined'))
    [tfms] = fn_calculate_TFM_for_all_focal_laws(filtered_data,exp_data,options_with_precalcs.focal_laws,options_with_precalcs.use_gpu_if_available,int_opt);
    sample_result=tfms.data;
    data.normalisation_factor = max(abs(tfms.norm_data(:)));
else
    [tfms] = fn_calculate_TFM_for_all_focal_laws(filtered_data,exp_data,options_with_precalcs.focal_laws,options_with_precalcs.use_gpu_if_available,int_opt);
    sample_result=tfms.data;

    % Get TFM normalisation factor from L-L backwall
    [tfms2] = fn_calculate_TFM_for_all_focal_laws(filtered_data,exp_data,options_with_precalcs.focal_laws_norm_region,options_with_precalcs.use_gpu_if_available,int_opt);
    data.normalisation_factor = max(abs(tfms2.data(:)));
end

sample_result=sample_result./data.normalisation_factor;
disp(['TFM time: ',num2str(toc(tstart2)),' seconds'])

nx=length(options_with_precalcs.data.x);
nz=length(options_with_precalcs.data.z);
%nviews=size(sample_result,2);

if (strcmp(options_with_precalcs.noise_levelling_status,'Active')>0)
    if (~isfield(options_with_precalcs,'mask') || ~isfield(options_with_precalcs,'rms'))
        options_with_precalcs.noise_levelling_status='Calculate';
    elseif (size(sample_result,1) ~= size(options_with_precalcs.mask,1) || size(sample_result,2) ~= size(options_with_precalcs.mask,2))
        disp('Recalculating mask/sigma parameters')
        options_with_precalcs.noise_levelling_status='Calculate';
    elseif options_with_precalcs.hole_fill ~= options_with_precalcs.used_hole_fill || options_with_precalcs.hole_expand ~= options_with_precalcs.used_hole_expand  || (options_with_precalcs.used_nremovals ~=options_with_precalcs.mask_node_limit && options_with_precalcs.mask_node_limit>0)
        disp('Recalculating mask/sigma parameters')
        options_with_precalcs.noise_levelling_status='Calculate';
    end
end

keep_old=0;
if (strcmp(options_with_precalcs.noise_levelling_status,'Calculate (Append)')>0)
    options_with_precalcs.noise_levelling_status='Calculate';
    keep_old=1;
end

switch options_with_precalcs.noise_levelling_status
case 'Off'
    % do nothing
case 'Calculate'
    % need to work out noise variables after adjusting TFM images for attenuation
    options_with_precalcs.levelling_changed=1;
    sample_result_abs=double(abs(sample_result));
    %sigma1=zeros(1,nviews);
    xval=zeros(1,nviews);xdash_lower=zeros(1,nviews);xdash_upper=zeros(1,nviews);
    % HICS removal
    view_psf=15*options_with_precalcs.pixel_size*1e-3; %Uses a
    nRemovals=max(10,round(0.02*(nz*nx))); perc_search=99;
    if (options_with_precalcs.mask_node_limit>0)
        nRemovals=options_with_precalcs.mask_node_limit;
    end
    options_with_precalcs.used_nremovals=nRemovals;
    options_with_precalcs.used_hole_fill=options_with_precalcs.hole_fill;
    options_with_precalcs.used_hole_expand=options_with_precalcs.hole_expand;
    
    options_with_precalcs.mask=zeros(size(sample_result_abs));
    %noise.nonmasked_counter=zeros(nviews,1);
    %noise.mask=zeros(size(sample_result_abs));
    
    noise.data=sample_result_abs./options_with_precalcs.attenMaps.data;
    mesh.x=data.x; mesh.z=data.z; text_output=0;
    [sigma1,noise.mask]=fn_masking_and_noise_characterisation2(mesh,noise.data,perc_search,nRemovals,options_with_precalcs.hole_fill,options_with_precalcs.hole_expand,text_output);
    
    for iview=1:nviews
        tmp_mask=find(~isnan(noise.mask(:,iview)));
        tmp_data=noise.data(tmp_mask,iview);
        %sigma1(iview)=fn_rayleigh_mle(tmp_data(:));
        [~,~,xval(iview),xdash_lower(iview),xdash_upper(iview)]=fn_calc_probplot_with_bounds(sigma1(iview),tmp_data(:),0,210);
    end 
    
    if (isfield(options_with_precalcs,'noise') && keep_old>0)
        options_with_precalcs.noise=[options_with_precalcs.noise noise];
        
        % need to update sigma and mask using combined data
        disp('Combining noise data, using conservative mask')
        options_with_precalcs.mask=ones(size(noise.mask));
        ndata=length(options_with_precalcs.noise);
        for i=1:ndata
            options_with_precalcs.mask=max(options_with_precalcs.noise(i).mask,options_with_precalcs.mask,'includenan');
        end
        % New sigma required, from all underlying data that hasn't been masked (using combined mask)
        for iview=1:nviews
            tmp_mask=find(~isnan(options_with_precalcs.mask(:,iview)));
            tmp_data=zeros(length(tmp_mask),ndata);
            for i=1:ndata
                tmp_data(:,i)=options_with_precalcs.noise(i).data(tmp_mask,iview);
            end
            sigma1(iview)=fn_rayleigh_mle(tmp_data(:));
            [~,~,xval(iview),xdash_lower(iview),xdash_upper(iview)]=fn_calc_probplot_with_bounds(sigma1(iview),tmp_data(:),0,210);
        end    
        
    else
        options_with_precalcs.noise=noise;
        options_with_precalcs.mask=noise.mask;
    end
    
    
    options_with_precalcs.b_fit=20*log10(xdash_upper./xdash_lower);
    options_with_precalcs.safety_factor=max(1,xdash_upper./xval);
    %options_with_precalcs.safety_factor
    options_with_precalcs.rms=sqrt(2).*sigma1.*options_with_precalcs.safety_factor; % Store RMS value of Rayleigh Noise
    %
    options_with_precalcs.noise_levelling_status='Active';
    %disp('End calculation phase')
case 'Active'
    % will be processed below   
end

if (isfield(options_with_precalcs,'b_fit'))
    data.b_fit=options_with_precalcs.b_fit;
    data.safety_factor=options_with_precalcs.safety_factor;
end
data.data=sample_result;

if (strcmp(options_with_precalcs.noise_levelling_status,'Active')>0)
    % Correct for attenuation
    data.noise_levelling_correction=1.0./options_with_precalcs.attenMaps.data;
    % Make relative to noise in TFM Image
    for iview=1:nviews
        data.noise_levelling_correction(:,iview)=data.noise_levelling_correction(:,iview)./(options_with_precalcs.rms(iview));
    end
    data.noise_levelling_status='Active';
    
    % Apply Masking (is now applied in plot_panel)
    % sample_result=sample_result./(options_with_precalcs.mask);
    % Store values associated with these modified TFM Images
    
    data.mask=options_with_precalcs.mask;
    data.rms=options_with_precalcs.rms;
    data.attenMaps=options_with_precalcs.attenMaps.data;
end 

data.masking_function=@fn_data_combining;
 
nwidth=options_with_precalcs.n_plots;
mheight=options_with_precalcs.m_plots;
gap=options_with_precalcs.gap;

data.m_plots=mheight;
data.n_plots=nwidth;
data.gap=gap;

%mheight=ceil(options_with_precalcs.sample_focal_law.views/nwidth);

full_width=nwidth*length(data.x)+(nwidth-1)*gap;
full_height=mheight*length(data.z)+(mheight-1)*gap;

data.f=NaN(full_height,full_width);
iwidth=0; iheight=1;
for j=1:nviews
    iwidth=iwidth+1;
    if (iwidth>nwidth)
        iwidth=1; iheight=iheight+1;
    end
    tmp=data.data(:,j);
    tmp=reshape(tmp,nz,nx);
    data.f((iheight-1)*nz+(iheight-1)*gap+1:iheight*nz+(iheight-1)*gap,(iwidth-1)*nx+1+(iwidth-1)*gap:iwidth*nx+(iwidth-1)*gap)=tmp;
end
data.display_type='TFM';
for j=1:nwidth
    data.xspacing(1,j)=(j-1)*(nx+gap);
    data.xspacing(2,j)=(j-1)*(nx+gap)+nx;
end
for j=1:iheight
    data.zspacing(1,j)=(j-1)*(nz+gap);
    data.zspacing(2,j)=(j-1)*(nz+gap)+nz;
end

data.xfull=full_width;
data.zfull=full_height;
data.xspacing=data.xspacing/data.xspacing(2,end);
data.zspacing=data.zspacing/data.zspacing(2,end);
%if ~options_with_precalcs.show_couplant_only
    %merge couplant and sample images
%     sample_pts = sum(options_with_precalcs.sample_focal_law.lookup_amp, 3) > 0;
%     sample_result = sample_result .* sample_pts;
%     couplant_result = options_with_precalcs.couplant_result .* (1 - sample_pts);
%     couplant_result = couplant_result / max(max(max(abs(couplant_result)))) * max(max(max(abs(sample_result))));
    %data.f = sample_result; % + couplant_result;
%else
%    data.f = options_with_precalcs.couplant_result;
%end

data.geom = options_with_precalcs.geom;
%disp(['Processing time: ',num2str(toc(tstart)),' seconds'])
%disp('End Processing')


end

%--------------------------------------------------------------------------

function info = fn_return_info(exp_data)
if ~isempty(exp_data) && any(exp_data.array.el_yc)
    info.fn_display = @gui_3d_plot_panel;
    info.display_options.interpolation = 0;
    no_pixels = 30;
else
    info.fn_display = @gui_2d_plot_panel;
    info.display_options.interpolation = 0;
    no_pixels = 100;
end
info.display_options.axis_equal = 1;
info.display_options.x_axis_sf = 1e3;
info.display_options.y_axis_sf = 1e3;
info.display_options.z_axis_sf = 1e3;
if isempty(exp_data)
    varargout{1} = [];
    varargout{2} = info;
    return %this is the exit point if exp_data does not exist
end

im_sz_z = min(27e-3,max(exp_data.time) * exp_data.ph_velocity / 2);
im_sz_xy = max([max(exp_data.array.el_xc) - min(exp_data.array.el_xc), ...
    max(exp_data.array.el_yc) - min(exp_data.array.el_yc)]);
info.options_info.x_size.label = 'X size (mm)';
info.options_info.x_size.default = 30e-3;
info.options_info.x_size.type = 'double';
info.options_info.x_size.constraint = [1e-3, 10];
info.options_info.x_size.multiplier = 1e-3;

info.options_info.x_offset.label = 'X offset (mm)';
info.options_info.x_offset.default = im_sz_xy;
info.options_info.x_offset.type = 'double';
info.options_info.x_offset.constraint = [-10, 10];
info.options_info.x_offset.multiplier = 1e-3;

if any(exp_data.array.el_yc)
    info.options_info.y_size.label = 'Y size (mm)';
    info.options_info.y_size.default = im_sz_xy;
    info.options_info.y_size.type = 'double';
    info.options_info.y_size.constraint = [1e-3, 10];
    info.options_info.y_size.multiplier = 1e-3;
    
    info.options_info.y_offset.label = 'Y offset (mm)';
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
info.options_info.z_offset.default = 0; %this should be calculated!
info.options_info.z_offset.type = 'double';
info.options_info.z_offset.constraint = [-10, 10];
info.options_info.z_offset.multiplier = 1e-3;

info.options_info.pixel_size.label = 'Pixel size (mm)';
info.options_info.pixel_size.default = max(0.5e-3,round(max([im_sz_xy, im_sz_z]) / no_pixels * 1e3*4)/(1e3*4));
info.options_info.pixel_size.default = min(info.options_info.pixel_size.default,0.3e-3);
info.options_info.pixel_size.type = 'double';
info.options_info.pixel_size.constraint = [1e-6, 1];
info.options_info.pixel_size.multiplier = 1e-3;

% info.options_info.array_standoff.label = 'Array standoff (mm)';
% info.options_info.array_standoff.default = im_sz_xy / 2;
% info.options_info.array_standoff.type = 'double';
% info.options_info.array_standoff.constraint = [1e-6, 1];
% info.options_info.array_standoff.multiplier = 1e-3;
% 
% info.options_info.array_inc_angle.label = 'Array incident angle (degs)';
% info.options_info.array_inc_angle.default = 0;
% info.options_info.array_inc_angle.type = 'double';
% info.options_info.array_inc_angle.constraint = [-90, 90] * pi / 180;
% info.options_info.array_inc_angle.multiplier = pi / 180;

% info.options_info.show_couplant_only.label = 'Show couplant only';
% info.options_info.show_couplant_only.type = 'bool';
% info.options_info.show_couplant_only.constraint = {'On', 'Off'};
% info.options_info.show_couplant_only.default = 0;

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
info.options_info.frac_half_bandwidth.default = 1.8;
info.options_info.frac_half_bandwidth.type = 'double';
info.options_info.frac_half_bandwidth.constraint = [0.01, 10];
info.options_info.frac_half_bandwidth.multiplier = 0.01;

tmp1=500;
if isfield(exp_data,'instrument_delay')
    tmp1=exp_data.instrument_delay*1e9;
end
info.options_info.instrument_delay.label = 'Inst. Delay (ns)';
info.options_info.instrument_delay.default = tmp1;
info.options_info.instrument_delay.type = 'double';
info.options_info.instrument_delay.constraint = [0, 20000];
info.options_info.instrument_delay.multiplier = 1;

tmp1=1480;
if isfield(exp_data,'water_velocity')
    tmp1=exp_data.water_velocity;
end
info.options_info.couplant_velocity.label = 'Couplant Velocity (m/s)';
info.options_info.couplant_velocity.default = tmp1;
info.options_info.couplant_velocity.type = 'double';
info.options_info.couplant_velocity.constraint = [1,20000];
info.options_info.couplant_velocity.multiplier = 1;

tmp1=25e-3;
if isfield(exp_data,'material_thickness')
    tmp1=exp_data.material_thickness;
end
info.options_info.backwall_depth.label = 'Mat. Thickness (mm)';
info.options_info.backwall_depth.default = tmp1;
info.options_info.backwall_depth.type = 'double';
info.options_info.backwall_depth.constraint = [0, 1e6];
info.options_info.backwall_depth.multiplier = 1e-3;

tmp1=exp_data.ph_velocity
if isfield(exp_data,'material_L_velocity')
    tmp1=exp_data.material_L_velocity;
end
info.options_info.ph_velocity.label = 'L Velocity (m/s)';
info.options_info.ph_velocity.default = tmp1;
info.options_info.ph_velocity.type = 'double';
info.options_info.ph_velocity.constraint = [1, 20000];
info.options_info.ph_velocity.multiplier = 1;

tmp1=exp_data.ph_velocity*0.5;
if isfield(exp_data,'material_T_velocity')
    tmp1=exp_data.material_T_velocity;
end
info.options_info.ph_velocity2.label = 'T Velocity (m/s)';
info.options_info.ph_velocity2.default = tmp1;
info.options_info.ph_velocity2.type = 'double';
info.options_info.ph_velocity2.constraint = [1, 20000];
info.options_info.ph_velocity2.multiplier = 1;

info.options_info.interpolation_method.label = 'Interpolation';
info.options_info.interpolation_method.default = 'Lanczos (a=2)';
info.options_info.interpolation_method.type = 'constrained';
info.options_info.interpolation_method.constraint = {'Linear', 'Nearest','Lanczos (a=2)','Lanczos (a=3)'};

%All about the surface
info.options_info.surface_pts_per_sample_wavelength.label = 'Surface pts/lambda';
info.options_info.surface_pts_per_sample_wavelength.default = 5;
info.options_info.surface_pts_per_sample_wavelength.type = 'double';
info.options_info.surface_pts_per_sample_wavelength.constraint = [0.01, 1000];
info.options_info.surface_pts_per_sample_wavelength.multiplier = 1;

tmp1='|M|easured';
if (isfield(exp_data,'location') && isfield(exp_data.location,'standoff')&& isfield(exp_data.location,'angle1'))
    tmp1='|S|pecified';
end
info.options_info.surface_type.label = 'Probe Location';
info.options_info.surface_type.default = tmp1;
info.options_info.surface_type.type = 'constrained';
info.options_info.surface_type.constraint = {'|S|pecified', '|M|easured'};

info.options_info.min_t.label = '|M| T Window Start (us)';
info.options_info.min_t.default = 10;
info.options_info.min_t.type = 'double';
info.options_info.min_t.constraint = [0,100000];
info.options_info.min_t.multiplier = 1;

info.options_info.max_t.label = '|M| T Window End (us)';
info.options_info.max_t.default = 100000;
info.options_info.max_t.type = 'double';
info.options_info.max_t.constraint = [0,100000];
info.options_info.max_t.multiplier = 1;

% Current measured values
tmp1=0;
if (isfield(exp_data,'location') && isfield(exp_data.location,'standoff'))
    tmp1=exp_data.location.standoff;
end
info.options_info.measured_probe_standoff.label = 'Standoff (mm)';
info.options_info.measured_probe_standoff.default = tmp1;
info.options_info.measured_probe_standoff.type = 'double';
info.options_info.measured_probe_standoff.constraint = [-1e6, 1e6];
info.options_info.measured_probe_standoff.multiplier = 1e-3;

tmp1=0;
if (isfield(exp_data,'location') && isfield(exp_data.location,'angle1'))
    tmp1=exp_data.location.angle1;
end
info.options_info.measured_probe_angle1.label = 'Angle (degrees)';
info.options_info.measured_probe_angle1.default = tmp1;
info.options_info.measured_probe_angle1.type = 'double';
info.options_info.measured_probe_angle1.constraint = [-360, 360];
info.options_info.measured_probe_angle1.multiplier = 1;

info.options_info.view_start.label = 'View(s) Start';
info.options_info.view_start.default = 1;
info.options_info.view_start.type = 'double';
info.options_info.view_start.constraint = [0,21];
info.options_info.view_start.multiplier = 1;

info.options_info.view_end.label = 'View(s) End';
info.options_info.view_end.default = 3;
info.options_info.view_end.type = 'double';
info.options_info.view_end.constraint = [0,21];
info.options_info.view_end.multiplier = 1;

info.options_info.noise_levelling_status.label = 'Noise Levelling';
info.options_info.noise_levelling_status.type = 'constrained';
info.options_info.noise_levelling_status.constraint = {'Active', 'Calculate','Calculate (Append)','Off'};
info.options_info.noise_levelling_status.default = 'Calculate';

%info.options_info.mask_from_file_status.label = 'Noise Mask & Data from File ';
%info.options_info.mask_from_file_status.type = 'constrained';
%info.options_info.mask_from_file_status.constraint = {'True', 'False'};
%info.options_info.mask_from_file_status.default = 'False';

info.options_info.attenuation_L.label = 'L Attenuation (Np/m)';
info.options_info.attenuation_L.default = 10.0;
info.options_info.attenuation_L.type = 'double';
info.options_info.attenuation_L.constraint = [0,100.0];
info.options_info.attenuation_L.multiplier = 1;

info.options_info.attenuation_T.label = 'T Attenuation (Np/m)';
info.options_info.attenuation_T.default = 30.0;
info.options_info.attenuation_T.type = 'double';
info.options_info.attenuation_T.constraint = [0,100.0];
info.options_info.attenuation_T.multiplier = 1;

info.options_info.mask_node_limit.label = 'Mask: Node Limit per Iter.';
info.options_info.mask_node_limit.default = 0;
info.options_info.mask_node_limit.type = 'double';
info.options_info.mask_node_limit.constraint = [0,100000];
info.options_info.mask_node_limit.multiplier = 1;

info.options_info.hole_fill.label = 'Mask: Clip Below Size';
info.options_info.hole_fill.default = 10;
info.options_info.hole_fill.type = 'double';
info.options_info.hole_fill.constraint = [0,100.0];
info.options_info.hole_fill.multiplier = 1;

info.options_info.hole_expand.label = 'Mask: Expansion Stage';
info.options_info.hole_expand.default = 3;
info.options_info.hole_expand.type = 'double';
info.options_info.hole_expand.constraint = [0,100.0];
info.options_info.hole_expand.multiplier = 1;

end

function [options]=fn_viewnames_active(options)
    if (options.value == 1)
        options.viewnames_active=1;
    else
        options.viewnames_active=0;
    end
end

function [options]=fn_mask_active(options)
    if (options.value == 1)
        options.masking_active=1;
    else
        options.masking_active=0;
    end
end

function [options]=fn_tfm_active(options)
    options.pvalues_active=0;
    options.tfm_active=1;
    options.levelling_active=0;
end

function [options]=fn_levelling_active(options)
    options.pvalues_active=0;
    options.tfm_active=0;
    options.levelling_active=1;
end

function [options]=fn_pvalues_active(options)
    options.pvalues_active=1;
    options.tfm_active=0;
    options.levelling_active=0;
    if (isfield(options,'pvalues_combined_active') && options.pvalues_combined_active>0)
        options.pvalues_active=2;
    end
end


function [enabled]=fn_button_mask_enabled(data,options)
    if isfield(data,'mask')
        enabled='on';
    else
        enabled='off';
    end
end

function [options]=fn_pvalues_combined_active(options)
    if (options.value == 1)
        options.pvalues_combined_active=1;
        if (options.pvalues_active >0)
            options.pvalues_active=2;
        end
    else
        options.pvalues_combined_active=0;
        if (options.pvalues_active>1)
            options.pvalues_active=1;
        end
    end
end

function [enabled]=fn_pvalues_combined_enabled(data,options)
    if isfield(options,'pvalues_active') && options.pvalues_active>0
        enabled='on';
    else
        enabled='off';
    end
end


function [data]=fn_data_combining(data,options)
% This function combines the multiview TFMs into a single image for plotting
% A small gap (consisting of NaN entries) is used to frame the subplots

mheight=data.m_plots;
nwidth=data.n_plots;
gap=data.gap;
nx=length(data.x);
nz=length(data.z);
full_width=data.xfull;
full_height=data.zfull;
data.f=NaN(full_height,full_width);
if (isfield(options,'pvalues_active'))
    data.pvalues_active=options.pvalues_active;
end
if (isfield(options,'levelling_active') && options.levelling_active>0)
    % Correct TFM values using attenuation based variation throughout the image view, to "level" the image to a common base
    sample_result=data.data.*data.noise_levelling_correction;
    data.display_type='Levelled';
elseif (isfield(options,'tfm_active') && options.tfm_active>0)
    % No levelling step, use the raw TFM image data
    sample_result=data.data;
    data.display_type='TFM';
else
    % Convert to p-values
    for i=1:size(data.attenMaps,2)
        cur_correction=1.0./double(data.attenMaps(:,i));
        x=abs(double(data.data(:,i)).*cur_correction); % Uses corrected TFM data
        current_sigma=double(data.rms(i))/sqrt(2);
        sample_result(:,i)=exp(-x.^2/(2.0*current_sigma^2)); %to check levelled values are correct: 20*log10(x./(current_sigma*sqrt(2))); %
    end
    data.display_type='P-values';
end

iwidth=0; iheight=1;
if (isfield(data,'normalisation_factor'))
    data.fmax=1.0;
else
    data.fmax=max(abs(sample_result(:)));
end
if (isfield(options,'pvalues_active') && options.pvalues_active>1)
    npixels=nx*nz;
    combine_type='P_OR';
    nviews=size(data.attenMaps,2);
    pvalue_data.combined=zeros(npixels,1);
    active_view_list=ones(nviews,1);
    pvalue_data.active_view_list=active_view_list;
    switch combine_type
    case 'P_OR'
        sFisher=zeros(npixels,1);
    case 'P_AND'
        sAND=zeros(npixels,1);
    end
        
    counter=zeros(npixels,1);
    for iview=1:nviews
        %Skip view if not selected
        if (active_view_list(iview)<1)
            continue;
        end
        %disp(['View ',num2str(iview)])
        if (options.masking_active<1)
            passed=1:npixels;
        else
            passed=find(data.mask(:,iview)>0);
        end
        switch combine_type
        case 'P_OR'
            sFisher(passed)=sFisher(passed)+log(sample_result(passed,iview));
        case 'P_AND'
            sAND(passed)=sAND(passed)+log(1-sample_result(passed,iview));
        end
        counter(passed)=counter(passed)+1;
    end
    nActive=sum(active_view_list);
    switch combine_type
    case 'P_OR'
    sFisher=-2*sFisher;
    case 'P_AND'
    sAND=-2*sAND;
    end
    if (nActive == 1)
        [~,iview]=max(active_view_list);
        pvalue_data.combined=sample_result(:,iview);
    else
        for i=1:nviews
            passed=find(counter == i);
            switch combine_type
            case 'P_OR'
                pvalue_data.combined(passed)=chi2cdf(sFisher(passed),2*i,'upper'); % NOTE: upper gives 1 - chi2cdf()
            case 'P_AND'
                pvalue_data.combined(passed)=1-chi2cdf(sAND(passed),2*i,'upper'); % NOTE: upper gives 1 - chi2cdf()
            end
        end
    end
    data.f2=reshape(pvalue_data.combined,nz,nx);
else
    for j=1:length(data.view_names)
        iwidth=iwidth+1;
        if (iwidth>nwidth)
            iwidth=1; iheight=iheight+1;
        end
        tmp=sample_result(:,j);
        tmp=reshape(tmp,nz,nx);
        if (isfield(options,'masking_active') && options.masking_active>0)
            %disp('Masking Active')
            tmp2=data.mask(:,j);
            tmp2=reshape(tmp2,nz,nx);
            data.f((iheight-1)*nz+(iheight-1)*gap+1:iheight*nz+(iheight-1)*gap,(iwidth-1)*nx+1+(iwidth-1)*gap:iwidth*nx+(iwidth-1)*gap)=tmp./tmp2;
        else
            %disp('Masking Inactive')
            data.f((iheight-1)*nz+(iheight-1)*gap+1:iheight*nz+(iheight-1)*gap,(iwidth-1)*nx+1+(iwidth-1)*gap:iwidth*nx+(iwidth-1)*gap)=tmp;
        end
         
    end
end
end

