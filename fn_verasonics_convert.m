function [exp_data]=fn_verasonics_convert(Trans, Receive, RcvData);
%convert verasonics data for BRAIN

new_data.ph_velocity=Trans.speedOfSound;
new_data.input_freq=Trans.frequency.*1e6;

rec_length=Receive(1).endSample-Receive(1).startSample+1;
num_els=Trans.numelements;

for ii=1:length(Receive)
   frames(ii)= Receive(ii).framenum;
end

els=[1:num_els];
[rx,tx]=meshgrid(els,els);

data=zeros(rec_length,num_els,num_els,max(frames));

n=1;
for ii=1:length(Receive) %load the first frame of data
    framenum = Receive(ii).framenum;
    data(:,:,n,framenum)=RcvData{1}(Receive(ii).startSample:Receive(ii).endSample,:,framenum);  
    if n==num_els
        n=0;
    end
    n=n+1;
end
dt=1./((Receive(1).decimSampleRate./Receive(1).quadDecim).*1e6);
%dt=1./(Receive(1).decimSampleRate*1e6);
lambda=new_data.ph_velocity/(Trans.frequency*1e6);

first_frame=data(:,:,:,1);

new_data.time_data=reshape(first_frame,size(first_frame,1),size(first_frame,2)*size(first_frame,3))./2^14;
new_data.tx=tx(:)';
new_data.rx=rx(:)';
start_time=(Receive(1).startDepth*lambda)/new_data.ph_velocity*2;
end_time=(Receive(1).endDepth*lambda)/new_data.ph_velocity*2;
new_data.time=[start_time:dt:end_time-dt]';

%convert the array definition
array.no_elements=num_els;
array.el_width=Trans.elementWidth*1e-3;
array.el_length=Trans.elevationApertureMm*1e-3;
array.el_pitch=Trans.spacingMm*1e-3;
array.el_type='rectangular';
array.element_numbers=[1:num_els];
array.el_tx=ones(1,num_els);
array.el_rx=ones(1,num_els);
array.el_xc=Trans.ElementPos(:,1)'*1e-3;
array.el_yc=Trans.ElementPos(:,2)'*1e-3;
array.el_zc=Trans.ElementPos(:,3)'*1e-3;
array.el_x1=zeros(1,num_els);
array.el_y1=zeros(1,num_els);
array.el_z1=zeros(1,num_els);
array.el_x2=zeros(1,num_els);
array.el_y2=zeros(1,num_els);
array.el_z2=zeros(1,num_els);
array.length=range(Trans.ElementPos(:,1).*1e-3);
array.centre_freq=Trans.frequency.*1e6;

new_data.array=array;

exp_data=new_data;
end