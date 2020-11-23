function [time_frontwall_PE,probe]=fn_extract_frontwall_signal_and_location(exp_data,instrument_delay,tstart,tend,filter_opt,half_bandwidth,filter_centre_freq)

%dimensions of probe
if (max(exp_data.array.el_yc) - min(exp_data.array.el_yc)>1.0e-10)
    dimension_opt=3;
else
    dimension_opt=2;
end

%time window signal
tstart2=0; tend2=0;
nsamples=length(exp_data.time);
for i = 1:nsamples
    if tstart2 == 0 && exp_data.time(i) > tstart 
        tstart2=i;
    end
    if exp_data.time(i) < tend 
        tend2=i;
    else
        break;
    end
end
if (tend2 <= tstart2)
    disp('Error in time windowing, t(end) < t(start)')
    return;
end

%% Use Pulse-Echo Only
if (size(exp_data.time_data,2)>length(exp_data.array.el_xc))
    exp_data.time_data=exp_data.time_data(:,exp_data.tx == exp_data.rx);
end

%% Filter data (Gaussian Frequency Filter, then absolute)
dexp = abs(fn_filter_time_data(filter_opt,exp_data,half_bandwidth,filter_centre_freq));

%% Extract max time value 
[~, maxloc]=max(dexp(tstart2:tend2,:));
maxloc=maxloc+tstart2-1;

interpolation_opt=1; % Use Lanczos Interpolation of A-Scan values, to seek maximum amplitude location
if (interpolation_opt<1)
    time_frontwall_PE=exp_data.time(maxloc);
else
    m1=max(1,maxloc-2);
    m2=min(nsamples,maxloc+2);
    time_frontwall_PE=zeros(length(exp_data.array.el_xc),1);
    t0=exp_data.time(1);
    dt=exp_data.time(2)-t0;
    for i=1:length(exp_data.array.el_xc)
        x=exp_data.time(m1(i):m2(i),1);
        %y=dexp(m1(i):m2(i),i);
        x2=x(1):0.01*(exp_data.time(2)-exp_data.time(1)):x(end);
        %y2=spline(x,y,x2);
        y3=zeros(size(x2));
        x3=(x2-t0)/dt+1;
        for j=1:length(x2)
            y3(j)=fn_lanczos_interpolation(dexp(:,i),x3(j) ,nsamples,3);
        end
        [~,mLoc]=max(y3);
        time_frontwall_PE(i)=x2(mLoc); 
    end
end

time_frontwall_PE=time_frontwall_PE(:);

%% Fit Line/Plane to data
dt=exp_data.time(2)-exp_data.time(1);
scaling=exp_data.ph_velocity/2.0;

if (dimension_opt>2)
    x=exp_data.array.el_xc;
    y=exp_data.array.el_yc;
    z=(time_frontwall_PE-instrument_delay)*scaling;
    [mx, my, c] = fn_calc_plane_through_point_cloud(x, y, z, 0);

    angleY=asin(mx)*180.0/pi;
    angleX=asin(my)*180.0/pi;

    z_offset=c;

    probe.dimensions=3;
    probe.standoff=z_offset;
    probe.angle1=angleX;
    probe.angle2=angleY;
else
    x=exp_data.array.el_xc.';
    %xlength=length(x);
    y=(time_frontwall_PE-instrument_delay)*scaling;
    
    [LR_offset,LR_angle]=fn_linear_line_best_fit(x,y,3*dt*scaling);
%     X = [ones(length(x),1) x];
%     b = X\y;
%     for ii=1:xlength
%         LR_offset=b(1);
%         LR_angle=asin(b(2))*180/pi;
%         y_calc=LR_offset+b(2)*x;  
%         y_calc_exceeded=find(abs(y_calc-y) > dt*3*scaling);
%         try
%             dummy=y_calc_exceeded(1);
%         catch
%             %disp(['No values exceeding error threshold in loop: ', int2str(ii)])
%             break;
%         end
%         y_mask=ones(size(y));
%         [~,qq2]=max(abs(y_calc-y));
%         y_mask(qq2)=0;
%         reduced=find(y_mask>0);
%         y_reduced=y(reduced);
%         y=y_reduced;
%         x_reduced=x(reduced);
%         x=x_reduced;
%         X = [ones(length(x),1) x];
%         b = X\y;
%     end
    probe.dimensions=2;
    probe.standoff=LR_offset;
    probe.angle1=LR_angle;
    probe.line_points_used=length(x);
    
    plot_on=0;
    if (plot_on>0)
       imagesc(exp_data.array.el_xc,exp_data.time,dexp)
       hold on
       plot(exp_data.array.el_xc,time_frontwall_PE,'-ro')
       plot(x,y/scaling+instrument_delay,'-go')
    end
end

end