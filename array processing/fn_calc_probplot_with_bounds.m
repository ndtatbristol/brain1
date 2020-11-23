function [mean_sigma0,stdev0,x,xdash_lower,xdash_upper,xx,ppM,grad0,grad1]=fn_calc_probplot_with_bounds(sigma00,xxx2,plot_on,varargin)

FCR_perc=0.1;
plot2_on=0; plot_text=1;
maxVal=0.99999;
minVal=0.01;
num=12; 
dist='rayleigh'; xlabel0='Data';
if (length(varargin)>0)
    num=varargin{1};
    if (length(varargin) > 1)
        plot_text=varargin{2};
    end
    if (length(varargin) > 2)
        FCR_perc=varargin{3};
    end
    if (length(varargin) >3)
        xlabel0=varargin{4};
    end
    if (length(varargin) >4)
        dist=varargin{5};
    end
    if (length(varargin) == 7)
        minVal=varargin{6};
        maxVal=varargin{7};
    end 
end
x_sorted=sort(xxx2);
[f,xx]=ecdf(x_sorted);

% d_step=1.0/length(x_sorted);
% f=f(2:end) - d_step/2;
% xx=xx(2:end);
% x25=prctile(xx,25);
% x75=prctile(xx,75);
% xIQRLine=[x25 x75];
% pIQRLine=[0.25 0.75];
% pIQRLine=sigma00*sqrt(-2*log(1-pIQRLine));
% gradIQRline=(pIQRLine(2)-pIQRLine(1))/(xIQRLine(2)-xIQRLine(1));
% cIQRline=-gradIQRline*xIQRLine(1)+pIQRLine(1);
%cdf0=1.0-exp(-xx(end)^2/(2*sigma00^2));
%f(end)=cdf0;


if (strcmp(dist,'rayleigh')>0)
    vals=[0.01 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.99 0.995 0.999 0.9995 0.9999];
    vals2=[vals 0.99999];
    pp0=sigma00*sqrt(-2*log(1-vals2));
elseif (strcmp(dist,'normal')>0)
    vals=[0.0001 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.99 0.999 0.9999];
    %vals2=[vals 0.99999];
    pp0=mean(xxx2)+std(xxx2)*sqrt(2)*erfinv(2*vals-1);
else
    disp('ERROR: unrecognised distribution')
end
if (plot_on>0)
    figure(num); clf; 
    
    ax = axes;
    
    
    if (strcmp(dist,'rayleigh')>0)
        plot(ax,pp0,pp0,'--k') 
        ax.YTick=sigma00*sqrt(-2*log(1-vals));
    elseif (strcmp(dist,'normal')>0)
        pp0=[min(xxx2) max(xxx2)];
        plot(ax,pp0,pp0,'--k') 
        ax.YTick=mean(xxx2)+std(xxx2)*sqrt(2)*erfinv(2*vals-1);
    end
    
    %ticLoc = get(ax,'YTick');
    ticLab = cellfun(@(x) num2str(x),num2cell(vals),'UniformOutput',false);
    set(ax,'YTickLabel',ticLab,'FontSize',14)
    %hold on
    %plot(ax,xIQRLine,pIQRLine,'--g');
    
end

%f(end)=0.99999;
if (strcmp(dist,'rayleigh')>0)
    ppM=sigma00*sqrt(-2*log(1-f));
elseif (strcmp(dist,'normal')>0)
    ppM=mean(xxx2)+std(xxx2)*sqrt(2)*erfinv(2*f-1);
end

if (plot_on>0)
    hold on
    plot(xx,ppM,'x');

    ylabel('Probability','FontSize',18)  
    xlabel(xlabel0)
    %axis equal
    axis tight
end

if (strcmp(dist,'normal')>0)
    return;
end

% size(xx)
% size(ppM)
pp=ppM; %sigma00*sqrt(-2*log(1-f));
w=find(xx>pp);
sf=ones(length(w),1);
for i=1:length(w)
    ip=w(i);
    if (f(ip)<minVal || f(ip)>maxVal)
        continue
    end
    %Expected
    expected=pp(ip);
    actual=xx(ip);
    %Safety factor required
    sf0=expected/actual;
    sf(i)=sf0;
end
grad0=min(sf);

w2=find(xx<pp);
sf2=zeros(length(w2),1);
for i=1:length(w2)
    ip=w2(i);
    if (f(ip)<minVal || f(ip)>maxVal)
        continue
    end
    %Expected
    expected=pp(ip);
    actual=xx(ip);
    %Safety factor required
    sf0=expected/actual;
    sf2(i)=sf0;
end
grad1=max(sf2);
aa=xx*grad0;
aa2=xx*grad1;

mean_sigma0=sigma00*sqrt(pi*0.5);
stdev0=sqrt((4-pi)*0.5*sigma00^2);
fcr_frac=1-(FCR_perc)/100.0;
x=double(sigma00*sqrt(-2*log(1-fcr_frac)));
xdash_lower=x/grad1;
xdash_upper=x/grad0;

if (plot_text<0)
    return
end

if (plot_on>0)
%     plot(xx,xx*gradIQRline+cIQRline,'--b');
    aa20=find(aa2<=max(pp0));
    plot(xx,aa,'--r');
    plot(xx(aa20),aa2(aa20),'--r');
end

% x=mean_sigma0;
% 
% if (plot_text>0)
%     x00=[0 x];
%     x01=[x x];
%     x02=[x 0];
%     if (plot_on>0)
%         q=[x00; x01; x02];
%         plot(q(:,1),q(:,2),'-.k')
%         txt1 = '  \mu ';
%         a=text(x00(1),x00(2),txt1);
%         a.FontSize=18;
%         a.VerticalAlignment='bottom';
%     end
% end

%mean_sigma_3stdev0=mean_sigma0+3*stdev0;


x00=[0 x];
x01=[x x];
x02=[x 0];
if (plot_on>0)
    q=[x00; x01; x02];
    if (plot_text>0)
        plot(q(:,1),q(:,2),'-.k')
        txt1 = ['  FCR = ',num2str(FCR_perc),'% '];
        if (plot_text>1)
            a=text(x00(1),x00(2),txt1);
            a.FontSize=18;
            a.VerticalAlignment='bottom';
            txt1 = ['\xi',char(10)];
            a=text(x02(1),x02(2),txt1);
            a.FontSize=18;
            a.VerticalAlignment='bottom';
            a.HorizontalAlignment='right';
        end
    end
end
%Min and Max cases

x01=[xdash_lower x];
x02=[xdash_lower 0];
if (plot_on>0)
    if (plot_text>0)
        q=[x01; x02];
        plot(q(:,1),q(:,2),'-.r')
        if (plot_text>1)
            txt1 = '\xi_L';
            a=text(x02(1),x02(2),txt1);
            a.FontSize=18;
            a.VerticalAlignment='bottom';
            a.HorizontalAlignment='right';
        end
    end
end

x01=[xdash_upper x];
x02=[xdash_upper 0];
if (plot_on>0)
    if (plot_text>0)
        q=[x01; x02];
        plot(q(:,1),q(:,2),'-.r')
        if (plot_text>1)
            txt1 = '\xi_U';
            a=text(x02(1),x02(2),txt1);
            a.FontSize=18;
            a.VerticalAlignment='bottom';
            a.HorizontalAlignment='left';
        end
    end
end
x01=[x x];
x02=[xdash_upper x];
if (plot_on>0 && plot_text>0)
    q=[x01; x02];
    plot(q(:,1),q(:,2),'-.k')
end
if (plot2_on>0&& plot_text>0)
    figure(11); clf;
plot(20*log10(pp(w)),sf,'xk');
end
xlim([0 max(xx)]);

end