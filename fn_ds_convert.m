function exp_data=fn_ds_convert(fname, varargin);
%load and convert diagnostic sonar datasets
%close all
%clear all

%fname='fit_frameFRD';
if nargin<2
    ph_vel=6300;
else
    ph_vel=varargin{1};
end

fnamepng=[fname '.png'];
fnamecfg=[fname '.cfg'];

%load and convert the ultrasonic data
data=imread(fnamepng);
data=double(data);
data=data-2^15;
data=data./2^15;
exp_data.time_data=double(data.');

%read in the configuration to set up the exp_data file correctly
delimiter = ' ';
startRow = 5;
endRow = inf;
formatSpec = '%s%s%s%s%s%s%s%s%[^\n\r]';
fileID = fopen(fnamecfg,'r');
textscan(fileID, '%[^\n\r]', startRow(1)-1, 'ReturnOnError', false);
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    textscan(fileID, '%[^\n\r]', startRow(block)-1, 'ReturnOnError', false);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'ReturnOnError', false);
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end
fclose(fileID);

sections={'[2D','[FMC','[Ascan]'};
need_fields=[4 6 3];

tot_count=1;
for jj=1:length(sections)
    count=1;
    found_field=0;
    while found_field==0
        found_field=strcmp(dataArray{1}{count},sections{jj});
        if found_field==1
            locs(jj)=count;
            for ii=1:need_fields(jj)
                cfg_dat{tot_count,1}=dataArray{1}{locs(jj)+ii};
                cfg_dat{tot_count,2}=dataArray{4}{locs(jj)+ii};
                tot_count=tot_count+1;
            end
        end
        count=count+1;
    end
end
num_els=str2num(cfg_dat{6,2});
samp_freq=str2num( cfg_dat{11,2})*1e6;
time_step=1/samp_freq;
time_len=str2num( cfg_dat{12,2});
start_samp=str2num( cfg_dat{13,2});
exp_data.time=[start_samp:(start_samp+time_len-1)].*time_step;
exp_data.time=exp_data.time.';

pitch=str2num( cfg_dat{3,2})*1e-3;
el_width=str2num( cfg_dat{4,2})*1e-3;
exp_data.array.centre_freq=str2num( cfg_dat{1,2})*1e6;
exp_data.ph_velocity=ph_vel;

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

count=0;
for ii=1:num_els
    for jj=1:num_els
        count=count+1;
        exp_data.tx(count)=ii;
        exp_data.rx(count)=jj;
    end
end

%save(fname,'exp_data')
end
