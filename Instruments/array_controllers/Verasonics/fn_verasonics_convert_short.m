function [new_data]=fn_verasonics_convert_short(Trans, Receive, RcvData);
%convert verasonics data for BRAIN
%disp('in convert')

new_data.ph_velocity=Trans.speedOfSound;
new_data.input_freq=Trans.frequency.*1e6;

rec_length=Receive(1).endSample-Receive(1).startSample+1;
num_els=Trans.numelements;

els=[1:num_els];
[rx,tx]=meshgrid(els,els);

receives=length(Receive);
for ii=1:receives
    acqs=Receive(ii).acqNum;
end
num_acqs=max(acqs);

data=zeros(rec_length,num_els,num_acqs);
for ii=1:num_acqs %load the first frame of data
    
    data(:,:,ii)=RcvData{1}(Receive(ii).startSample:Receive(ii).endSample,:);  
    
end
dt=1./((Receive(1).decimSampleRate./Receive(1).quadDecim).*1e6);
lambda=new_data.ph_velocity/(Trans.frequency*1e6);

if receives/num_acqs ==1
    new_data.time_data=(reshape(data,size(data,1),size(data,2)*size(data,3))./2^14);
else
    new_data.time_data=(reshape(data,size(data,1),size(data,2)*size(data,3))./2^15);
end
new_data.tx=tx(:)';
new_data.rx=rx(:)';
start_time=(Receive(1).startDepth*lambda)/new_data.ph_velocity*2;
end_time=(Receive(1).endDepth*lambda)/new_data.ph_velocity*2;
new_data.time=[start_time:dt:end_time-dt]';

int_fact=2;
data_len=length(new_data.time);
%new_data.time_data=interpft(new_data.time_data,data_len*int_fact);
%new_data.time=linspace(min(new_data.time),max(new_data.time),data_len*int_fact)';
%exp_data=new_data;
end