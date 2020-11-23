function [a,b]=fn_fit_ellipse(angle,vel);
dec_pla=4;
angle_rnd=round(angle.*10^dec_pla)./10^dec_pla;
un_ang=unique(angle_rnd);
for ii=1:length(un_ang)
    use_ind=find(angle_rnd==un_ang(ii));
    un_vel(ii)=mean(vel(use_ind));
end

angle=un_ang;
vel=un_vel;

in_a=interp1(angle,vel,0);
in_b=2900;

a=in_a;

min_ang=min(angle);
max_ang=max(angle);
use_angs = find(angle<max_ang & angle>min_ang);
angles=angle(use_angs);

%angles=linspace(min(angle),max(angle),100);
steps=linspace(0.8,2,200);
for ii=1:length(steps)
    b=in_b.*steps(ii);
    vels(:,ii)=(a.*b)./sqrt((b.*cos(angles)).^2+(a.*sin(angles)).^2);
    error(ii)=sqrt(sum((vel(use_angs)'-vels(:,ii)).^2)./length(steps));
end
[val loc]=min(error);
b=in_b.*steps(loc);
end