function exp_data=fn_m2m_convert(fname, varargin);

%==========================================================================
% civa2mat function:
% Function that converts FMC text files exported with CIVA into matlab
% matrices. This function works only for linear array data.
% The ouput data amplitudes are given as matrices in points or dB:
%    Hij    --> (Time x N_receiver x N_transmitter) in points
%    Hij_dB --> (Time x N_receiver x N_transmitter) in dB
%==========================================================================
% Auteur : S. SHAHJAHAN
% version : v1 (english/ only FMC conversion 24/01/2015)
% Dernière mise à jour : 17/06/2010
%==========================================================================
if nargin<2
    gap=11;
else
    gap=varargin{1};
end

fname=[fname '.txt'];
%[FILENAME, PATHNAME] = uigetfile('*.txt','Select the FMC data in .txt format');
chemin1 = char(fname);

fid1 = fopen(chemin1);
data1 = dlmread(char(chemin1),';',gap,0);

% Displaying the headlines
for ii = 1:gap
    str = fgetl(fid1);
    count=1;
    for jj=1:length(str)
       [x ok]=str2num(str(jj));
       if ok==1
         ok_loc(count)=jj;
         count=count+1;
       end
    end
    all_ok{ii}=ok_loc;
    allstr{ii}=str;
end
fclose(fid1);
fs=str2num(allstr{1}(all_ok{1}(find(diff(all_ok{1})==1,1)+1):all_ok{1}(end)))*1e6;
samples=str2num(allstr{2}(all_ok{2}(find(diff(all_ok{2})==1,1)+1):all_ok{2}(end)));
amp=str2num(allstr{3}(all_ok{3}(find(diff(all_ok{3})==1,1)+1):all_ok{3}(end)));
poss_els=all_ok{11}(end-4:end);
diff_poss_els=find(diff(poss_els)==1);
num_els=str2num(allstr{11}(poss_els(diff_poss_els(1)):poss_els(diff_poss_els(end))+1))+1;

[trans rec]=meshgrid([1:num_els],[1:num_els]);

J = num_els;
I = J;
N = samples;
% Extraction of the B-scans for each sequence of emission
ascan_brute1_dB = data1(:,3:J+2);    % contains amplitudes in dB
ascan_brute1 = data1(:,J+3:2*J+2);   % contains amplitudes in points
t = data1(1:N,2)*1e-6; % Time vector of the experiment
%clear data1;

% Memory allocation
Hij = zeros(N,J,I);
Hij_dB = zeros(N,J,I);

% A-scans are sorted into matrices
for i = 0:I-1
    idx(:,1) = (1+i*N):(N+i*N);
    Hij(1:N,:,i+1) = ascan_brute1(idx,:);
    Hij_dB(1:N,:,i+1) = ascan_brute1_dB(idx,:);
end

time_data=reshape(Hij,[N,J*I]);
trans=reshape(trans,[1,J*I]);
rec=reshape(rec,[1,J*I]);
exp_data.tx=trans;
exp_data.rx=rec;
exp_data.time=t;
exp_data.time_data=time_data./amp;

%make up some values for phase velocity and array parameters to keep brain
%happy
exp_data.ph_velocity=6300;
pitch=1.5*1e-3;
el_width=1.4*1e-3;
exp_data.array.centre_freq=2*1e6;
el_xc=[1:num_els]*pitch;

exp_data.array.el_xc=el_xc-mean(el_xc);
exp_data.array.el_x1=exp_data.array.el_xc-pitch/2;
exp_data.array.el_x1=exp_data.array.el_xc+pitch/2;
exp_data.array.el_yc=zeros(size(el_xc));
exp_data.array.el_y1=exp_data.array.el_yc;
exp_data.array.el_y2=exp_data.array.el_yc;
exp_data.array.el_zc=zeros(size(el_xc));
exp_data.array.el_z1=exp_data.array.el_zc;
exp_data.array.el_z2=exp_data.array.el_zc;


%save exp_data exp_data
end
