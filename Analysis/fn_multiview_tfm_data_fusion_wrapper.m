function info = fn_multiview_tfm_data_fusion_wrapper(exp_data, data, display_options, process_options, h_fn_set_process_options, h_fn_process_options_changed)
if isempty(exp_data) & isempty(data)
    info.name = 'TFM P-Values Data Fusion';
    return;
else
    info = [];
end;

if (~isfield(data,'rms'))
    info = [];
    disp('Input not suitable for data fusion')
    return;
end
    

%figure size
width = 800;
height = 480;

min_width=120/width;
table_width_frac=min(min_width,0.25);
table_height_frac=0.3;
button_height_frac=0.08;


status_height_pixels=30;
table_pos = [0,1-table_height_frac,table_width_frac,table_height_frac];
result_pos = [0,0,table_width_frac,1-table_height_frac-button_height_frac];
graph_pos = [table_width_frac,0,1-table_width_frac,1];
button_pos = [0,1-table_height_frac-button_height_frac,table_width_frac,button_height_frac];



%create figure
p = get(0, 'ScreenSize');
f = figure('Position',[(p(3) - width) / 2, (p(4) - height) / 2, width, height] ,...
    'MenuBar', 'none', ...
    'NumberTitle', 'off', ...
    'ToolBar', 'None', ...
    'Name', ['Analysis:', ' TFM Data Fusion'] ...
);

set(f, 'WindowButtonDownFcn', @MouseClick);
set(f, 'WindowButtonMotionFcn', @MouseClick);
%set(f, 'WindowScrollWheelFcn', @MouseScroll);
%set(f, 'KeyPressFcn', @KeyPress )

config = fn_get_config;
panel_colour = config.general.window_bg_color;
%load icons
icons = [];
load(config.files.icon_file);

colormap(f,flipud(jet));

%create graph panel
h_panel = uipanel(f, 'Units', 'Normalized', 'Position', graph_pos);

%redefine the resize function of the container panel
set(h_panel, 'ResizeFcn', @fn_resize);
h_graph_panel = uipanel('Parent', h_panel);
hcontrol2 = uipanel('Parent', h_panel, 'BackgroundColor', panel_colour);  

h_custom_button = uicontrol(hcontrol2, ...
                        'String', 'View Contributions Plot', ...
                        'Style', 'pushbutton', ...
                        'Callback', @cb_view_switch, ...
                        'Tag', 'custom','Position',[0 0 140 28]);
                        
h_save_button = uicontrol(hcontrol2, ...
                        'CData', fn_get_cdata_for_named_icon(icons, 'Standard.SaveFigure'), ...
                        'Style', 'pushbutton', ...
                        'Callback', @cb_save, ...
                        'TooltipString', 'Save data/image', ...
                        'HandleVisibility', 'Off', ...
                        'Tag', 'custom','Position',[140 0 28 28]);    
                        
plot_view_switch=1;
h_info_text = uicontrol(hcontrol2, 'Style', 'Text', 'String', {''}, 'HorizontalAlignment', 'Right', 'Units', 'Normalized', 'Position', [0.5, 0, 0.48, 1], 'BackgroundColor', panel_colour);
h_info_text2 = uicontrol(hcontrol2, 'Style', 'Text', 'String', {''}, 'HorizontalAlignment', 'Right', 'Units', 'Normalized', 'Position', [0.3, 0, 0.68, 0.5], 'BackgroundColor', panel_colour);
%if isfield(process_options, 'atten')
    %create update button
    h_update_button = uicontrol(f, 'Style', 'pushbutton', 'Units', 'Normalized', 'Position', button_pos, 'String', 'Select All Views', 'Enable', 'On', 'Callback', @cb_update);
%end

%create options table
[h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(f, table_pos, 'normalized', @fn_new_params);
% keyboard

content_info.combine_type.label = 'Fusion Type';
content_info.combine_type.default = 'P_OR';
content_info.combine_type.type = 'constrained';
content_info.combine_type.constraint = {'P_OR', 'P_AND'};

content_info.ignore_mask.label = 'Ignore Mask';
content_info.ignore_mask.default = 'False';
content_info.ignore_mask.type = 'constrained';
content_info.ignore_mask.constraint = {'True', 'False'};

content_info.color_max.label = 'P-Value (max)';
content_info.color_max.default = 0.01;
content_info.color_max.type = 'double';
content_info.color_max.multiplier = 1;
content_info.color_max.constraint = [0,100];

content_info.color_min.label = 'P-Value (min) ';
content_info.color_min.default = 0.0;
content_info.color_min.type = 'double';
content_info.color_min.multiplier = 1;
content_info.color_min.constraint = [-100,1];

h_fn_set_content(content_info);



%h_result = uicontrol(g, 'Units', 'Normalized', 'Position', result_pos);
h_list = uicontrol(f,'Style','list','max',10,'min',1,'Units', 'Normalized','Position',result_pos,'string',data.view_names,'Callback',@fn_new_params);
h_list.Value=1:length(data.view_names);   
a = axes('Parent', h_graph_panel);

pvalue_data=[];

fn_resize;

%trigger the calc
h_data_changed();
    
    function fn_new_params(varargin)
    
        if (nargin == 1)
            params=varargin{1};
        else
            params=h_fn_get_data();
        end
    
        if (strcmp(params.ignore_mask,'True')>0)
            ignore_mask=true;
        else
            ignore_mask=false;
        end
        if (~isfield(data,'mask'))
            disp('Input is not suitable for data fusion')
            return;
        end
        [pvalue_data] = fn_calc_pvalue(data,params.combine_type,ignore_mask);
        if (plot_view_switch>0)
            tmp=reshape(pvalue_data.combined,length(data.z),length(data.x));
        else
            tmp=reshape(pvalue_data.counter,length(data.z),length(data.x));
        end
        tmp2=ones(size(tmp));
        tmp2(pvalue_data.counter<1)=0;
        hh=imagesc(data.x* display_options.x_axis_sf,data.z*display_options.z_axis_sf,tmp);
        axis equal
        axis tight
        if (plot_view_switch>0)
            h_custom_button.String='View Contributions Plot';
            count=pvalue_data.counter>0;
            check_val=sum(pvalue_data.combined(count) < params.color_max)/sum(count);
            h_info_text2.String=['Lowest p-value: ',num2str(min(pvalue_data.combined(count))),'. Image fraction below p-value = ',num2str(params.color_max),' is ',num2str(check_val),''];
        else
            h_custom_button.String='View P-Values Plot';
            h_info_text2.String='';
        end
        set(hh,'alphadata',tmp2);
        if (plot_view_switch>0)
            try
                caxis([params.color_min params.color_max])
            catch
            
            end
        else
            try
                caxis([1 max(pvalue_data.counter(:))])
            catch
            
            end
        end
        colorbar
        hold on
        %frontwall & backwall on ploy
        line([data.x(1) data.x(end)] * display_options.x_axis_sf, [0 0] * display_options.z_axis_sf, ...
            'Color', 'r', ...
            'Tag', 'geom');
        line([data.x(1) data.x(end)] * display_options.x_axis_sf, [data.geom.backwall_z data.geom.backwall_z] * display_options.z_axis_sf, ...
        'Color', 'g', ...
        'Tag', 'geom');
        hold off
        

        %set(h_update_button, 'Enable', 'On');
    end

    function cb_update(a, b)
        %process_options.atten = process_options.atten - new_atten_db;
        all_selected=1:length(data.view_names);
        h_list.Value=all_selected;
        fn_new_params;
        %h_fn_set_process_options(process_options);
        %h_fn_process_options_changed(process_options);
    end


    function [pvalue_data] = fn_calc_pvalue(data,combine_type,ignore_mask);
    nviews=length(data.view_names);
    
    %Get current selected views, ignore other non-selected views
    active_view_list=zeros(nviews,1);
    active_view_list(h_list.Value)=1;
    
    npixels=size(data.data,1);
    pvalue_data.views=zeros(npixels,nviews);
    for i=1:nviews
        cur_correction=1.0./double(data.attenMaps(:,i));
        x=abs(double(data.data(:,i)).*cur_correction); % Uses corrected TFM data
        current_sigma=double(data.rms(i))/sqrt(2);
        pvalue_data.views(:,i)=exp(-x.^2/(2.0*current_sigma^2)); %to check levelled values are correct: 20*log10(x./(current_sigma*sqrt(2))); %
        % if (i == 1 || i==19)
            % passed=find(data.mask(:,i)>0);
            % x1=max(x(passed))
            % xExp=-x1.^2/(2.0*current_sigma^2)
            % e0=exp(xExp)
            % pp0=current_sigma*sqrt(-2*log(e0))
            % %e1=1e-17;
            % %pp0=current_sigma*sqrt(-2*log(e1))
            
            % e2=1-chi2cdf(-2*log(e0),2)
        % end
    end
    
    %disp(['Min individual p-value is ',num2str(min(pvalue_data.views(:)))])

    pvalue_data.combined=zeros(npixels,1);
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
        if (ignore_mask)
            passed=1:npixels;
        else
            passed=find(data.mask(:,iview)>0);
        end
        switch combine_type
        case 'P_OR'
            sFisher(passed)=sFisher(passed)+log(pvalue_data.views(passed,iview));
        case 'P_AND'
            sAND(passed)=sAND(passed)+log(1-pvalue_data.views(passed,iview));
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
        pvalue_data.combined=pvalue_data.views(:,iview);
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
    %passed=find(counter >0);
    % cur_correction=1.0./double(data.attenMaps(passed,1));
    % x=abs(double(data.data(passed,1)).*cur_correction); % Uses corrected TFM data
    % current_sigma=double(data.rms(1))/sqrt(2);
    % (-max(x(:)).^2/(2.0*current_sigma^2))
    % exp(-max(x(:)).^2/(2.0*current_sigma^2))
    % disp('here')
    % pvalue_data.combined(:)=pvalue_data.views(:,1);
    %disp(['Max disc ',num2str(max(abs(pvalue_data.combined(passed)-pvalue_data.views(passed,1))))])
    
    
    
    pvalue_data.counter=counter;    
    pvalue_data.combine_type=combine_type;
    
    
    end
    function fn_resize(src, evt)
    
        p = getpixelposition(h_panel);
        %         p(1:2) = p(1:2) + 1;
        %         p(3:4) = p(3:4) - 2;
        %setpixelposition(h_list, [p(3) - slider_width_pixels, status_height_pixels + 1, slider_width_pixels, p(4) - status_height_pixels]);
        setpixelposition(hcontrol2, [1, 1, p(3), status_height_pixels]);
        % if (isfield(options,'custom_button'))
            % control2_height_pixels=round(status_height_pixels*0.5);
            % setpixelposition(hcontrol2, [1, status_height_pixels + 1, p(3)- slider_width_pixels, control2_height_pixels]);
        % else
            % control2_height_pixels=0;
        % end
        %setpixelposition(h_panels.plot, [1, status_height_pixels + control2_height_pixels + 1, p(3) - slider_width_pixels, p(4) - status_height_pixels - control2_height_pixels]);
    
    end
    function MouseClick(object,eventdata)
        C = get (gca, 'CurrentPoint');
        if (C(1,1)<data.x(1)*display_options.x_axis_sf || C(1,1)>data.x(end)*display_options.x_axis_sf)
            h_info_text.String='';
            return;
        end 
        if (C(1,2)<data.z(1)*display_options.z_axis_sf || C(1,2)>data.z(end)*display_options.z_axis_sf)
            h_info_text.String='';
            return;
        end    
        C2(1,1)=C(1,1)/display_options.x_axis_sf;
        C2(1,2)=C(1,2)/display_options.z_axis_sf;
        dx=data.x(2)-data.x(1);
        C3(1,1)=round((C2(1,1)-data.x(1))/dx)+1;
        dz=data.z(2)-data.z(1);
        C3(1,2)=round((C2(1,2)-data.z(1))/dz)+1;
        C3=max(C3,1); C3(1,1)=min(C3(1,1),length(data.x));
        %pvalue_data
        
        ind1=sub2ind([length(data.z),length(data.x)],C3(1,2),C3(1,1));
        if (pvalue_data.counter(ind1) < 1)
            h_info_text.String='';
            return;
        end
        str=['x = ', num2str(data.x(C3(1,1))*display_options.x_axis_sf), ', z = ',num2str(data.z(C3(1,2))*display_options.x_axis_sf),' '];
        if (plot_view_switch>0)
            h_info_text.String=[str,'p-value = ',num2str(pvalue_data.combined(ind1)),' '];
        else
            h_info_text.String=[str,'contributions = ',num2str(pvalue_data.counter(ind1)),' '];
        end
    end
    function cb_view_switch(varargin)
        plot_view_switch=plot_view_switch+1;
        if (plot_view_switch>1) 
            plot_view_switch=0;
        end
        if (plot_view_switch>0)
            h_custom_button.String='View Contributions Plot';
        else
            h_custom_button.String='View P-Values Plot';
        end
        fn_new_params
    end
    function cb_save(a, b)
        filter{1,1} = '*.fig'; filter{1,2} = 'Matlab figure (*.fig)';
        filter{2,1} = '*.png'; filter{2,2} = 'Portable Network Graphics (*.png)';
        filter{3,1} = '*.jpg'; filter{3,2} = 'JPEG (*.jpg)';
        filter{4,1} = '*.eps'; filter{4,2} = 'EPS level 1 (*.eps)';
        filter{5,1} = '*.mat'; filter{5,2} = 'P-Values data (*.mat)';
        [fname, data_folder, filterindex] = uiputfile(filter, 'Save');
        if (fname == 0) % catch Save Dialog exiting without a specified filename (e.g. cancel button hit)
            disp('No filename specified. Skipping')
            return;
        end
        if filterindex < 5
            saveas(fig_handle, fullfile(data_folder, fname));
        else
            save(fullfile(data_folder, fname), 'data','pvalue_data','exp_data');
        end
    
    end
    
end