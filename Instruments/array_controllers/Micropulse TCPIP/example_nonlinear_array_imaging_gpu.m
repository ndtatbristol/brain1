%%Nonlinear array imaging demo

if exist('tcpip_obj') && isa(tcpip_obj, 'tcpip')
    fclose(tcpip_obj);
    delete(tcpip_obj);
    clear('tcpip_obj');
    disp('Clearing');
end

clear all;
close all;

global tcpip_obj


%--------------------------------settings for the exp---------------------%
%Array and material details - all NDT lab arrays should have a corresponding file in the
%NDT library!
array_fname = 'Imasonic 1D 64els 5.00MHz 0.63mm pitch.mat';

%mat_fname = 'Aluminium (6300).mat';
mat_fname = 'Steel (5850).mat';

%Micropulse acquisition details
half_matrix_capture = 0; %1 for half matrix or 0 for full matrix
test_options.sample_freq = 25e6; %can be 25MHz, 50MHz or 100MHz
test_options.pulse_width = 80e-9;
test_options.time_pts = 3000;
test_options.sample_bits = 16; %8, 10, 12, 16
test_options.filter_no = 2; %1:5-10MHz 2:2-10MHz 3:0.75-5MHz 4:0.75-20MHz
test_options.prf = 100; %maximum pulse repetition frequency
test_options.gate_start =20e-5;%gate start time in s

N=128; %focused capture number
no_buffs=2;  %number of padding captures

f_gain=45; %parallel gain 
fmc_gain= round(20*log10(8*10^(f_gain/20))); %sequential gain

f_volt=200; %pulse voltages
FMC_volt=f_volt;

equalise_averages=1;
software_averages=1;
focused_averages=1; %hardware averages
FMC_averages=focused_averages;

%Save data information if required
save_data = 0; % 1 to save ,0 not to save

%imaging properties
x_size =30e-3;
z_size = 30e-3;
x_offset = 0;
z_offset = 10e-3;%20e-3;%20e-3;
pixel_size = 1e-3;
centre_freq = 5e6;

%frequencies to use for NL image
lower_f=2/3*centre_freq;
upper_f=4/3*centre_freq;

%options for focal law generation
focal_law_options.angle_dep_vel = 0;
focal_law_options.angle_limit = 0; %zero angle limit removes the limit and includes all 180/180*pi;

%--------------------open acquisition hardware and setup tests------------%
%Load the array file and set field in exp_data
tmp = load(array_fname);
exp_data.array = tmp.array;

exp_data.num_els = length(exp_data.array.el_xc);

tmp = load(mat_fname);
exp_data.ph_velocity = tmp.material.ph_velocity;

if focal_law_options.angle_dep_vel
    exp_data.vel_poly = tmp.material.vel_poly;
end
exp_data.tx = ones(exp_data.num_els,1);
exp_data.rx = exp_data.tx;

data.x = [-x_size / 2: pixel_size: x_size / 2] + x_offset;
data.z = [0: pixel_size: z_size] + z_offset;

[mesh.x, mesh.z] = meshgrid(data.x, data.z);

exp_data.time=[1:test_options.time_pts]/test_options.sample_freq;

focal_law = fn_calc_tfm_focal_law2(exp_data, mesh, focal_law_options);

%set all amps to 1
amp_corrs=find(focal_law.lookup_amp~=0);
focal_law.lookup_amp(amp_corrs) = 1;

focal_law.maxs = max(focal_law.lookup_time,[],3);
focal_law.maxs = repmat(focal_law.maxs,[1 1 exp_data.num_els]);

focal_law.delay_law_array = focal_law.maxs - focal_law.lookup_time;
focal_law.delay_law_vec = reshape(focal_law.delay_law_array,numel(mesh.x),exp_data.num_els);
focal_law.amps = reshape(focal_law.lookup_amp,numel(mesh.x),exp_data.num_els);

freq_arr=((1:test_options.time_pts)-1)/test_options.time_pts*test_options.sample_freq;

lower_el=ceil(lower_f/test_options.sample_freq*test_options.time_pts);
upper_el=floor(upper_f/test_options.sample_freq*test_options.time_pts);

freq_arr_window=freq_arr(lower_el:upper_el);
freq_mat_window_sq=(ones(exp_data.num_els,1)*freq_arr_window.^2)';

window_pts=length(freq_arr_window);

max_offset_el=ceil(max(max(focal_law.delay_law_vec))*test_options.sample_freq);

%% -------------FMC capture---------------------------

test_options.averages =FMC_averages;
test_options.pulse_voltage = FMC_volt;
test_options.db_gain=fmc_gain;
half_matrix_capture = 0;

[test_options.tx_ch, test_options.rx_ch] = fn_set_fmc_input_matrices(exp_data.num_els, half_matrix_capture);
test_options.tx_ch=[ones(no_buffs, exp_data.num_els); test_options.tx_ch];
test_options.rx_ch=[ones(no_buffs, exp_data.num_els); test_options.rx_ch];

%Micropulse connection details
ip_address = '10.1.1.2';
port_no = 1067;
echo_on = 0;
%Micropulse reset details
full_reset = 1;

%Connect
if ~fn_ag_connect_tcpip(ip_address, port_no, 1)
    disp('Failed to connect');
    return;
end;

%Reset if required
if full_reset
    reset_result = fn_ag_reset_tcpip(echo_on);
end;

[exp_data.tx, exp_data.rx] = fn_ag_set_test_options_tcpip(test_options, echo_on);
pause(0.2)

full_mat=zeros(test_options.time_pts,  exp_data.num_els^2 + no_buffs*exp_data.num_els);
for kk=1:equalise_averages%software_averages %FMC acquisition
    disp(kk)
    exp_data.time_data = fn_ag_do_test_tcpip(echo_on);
    full_mat = full_mat+exp_data.time_data;
end
full_mat=full_mat/equalise_averages;%software_averages;
full_mat=full_mat(:, (no_buffs*exp_data.num_els +1):end);
test_options.tx_ch=test_options.tx_ch(no_buffs+1:end,:);
test_options.rx_ch=test_options.rx_ch(no_buffs+1:end,:);

disp(['Max fmc value: ' num2str(max(max(full_mat)))])

full_mat=full_mat*10^(f_gain/20)/10^(fmc_gain/20)*f_volt/FMC_volt; %normalising reception gains

for mm=1:exp_data.num_els
    full_matrix_data(:,:,mm)= full_mat(:,((mm-1)*exp_data.num_els+1):mm*exp_data.num_els);
end

if save_data
    save fmc_set full_matrix_data
end

%% Creating GPU arrays
full_matrix_data=gpuArray((full_matrix_data));
full_matrix_data=fft(full_matrix_data);

freq_mat_window_sq=gpuArray(freq_mat_window_sq);
freq_arr_window=gpuArray(freq_arr_window');
freq_arr=gpuArray(freq_arr');

gpuones=gpuArray(ones(1,exp_data.num_els));
focused_mat=gpuArray(zeros(test_options.time_pts,exp_data.num_els));
single_set=gpuArray(zeros(test_options.time_pts,exp_data.num_els));
gpuonesarr=gpuArray((1:test_options.time_pts)'-1);
freq_delay=gpuArray(ones(test_options.time_pts,1));

%% ------------------ Focused captures and data processing--------------------
test_options.averages = focused_averages;
test_options.pulse_voltage = f_volt;
test_options.db_gain = f_gain;
test_options.tx_ch=ones(N,exp_data.num_els);
test_options.rx_ch=ones(N,exp_data.num_els);
test_options.rx_delay_law_array=zeros(N,exp_data.num_els);

nonlinear_image_data=zeros(size(mesh.x));

for nn=1:ceil(numel(mesh.x)/N)
    
    if nn==ceil(numel(mesh.x)/N) && mod(numel(mesh.x), N)~=0
        Nt=mod(numel(mesh.x), N);
        test_options.tx_ch=ones(Nt,exp_data.num_els);
        test_options.rx_ch=ones(Nt,exp_data.num_els);
        test_options.rx_delay_law_array=zeros(Nt,exp_data.num_els);
    else
        Nt=N;
    end
    
    test_options.tx_ch = focal_law.amps(1+(nn-1)*N:(nn-1)*N+Nt,:);
    test_options.tx_delay_law_array=focal_law.delay_law_vec(1+(nn-1)*N:(nn-1)*N+Nt,:);
    
    %adding buffer captures to start
    test_options.tx_ch=[ones(no_buffs, exp_data.num_els); test_options.tx_ch];
    test_options.rx_ch=[ones(no_buffs, exp_data.num_els); test_options.rx_ch];
    test_options.tx_delay_law_array=[zeros(no_buffs, exp_data.num_els);test_options.tx_delay_law_array]; 
    test_options.rx_delay_law_array=[zeros(no_buffs ,exp_data.num_els); test_options.rx_delay_law_array];
    
    
    %Send detailed test_options to Micropulse
    [exp_data.tx, exp_data.rx] = fn_ag_set_test_options_tcpip(test_options, echo_on);
    
    exp_data.time_data=[];
    
    %Capture one set of data from Micropulse
    time_data_tmp=zeros( test_options.time_pts, size(test_options.rx_ch, 1)*size(test_options.rx_ch, 2));
    for kk=1:software_averages
        exp_data.time_data = fn_ag_do_test_tcpip(echo_on);
        time_data_tmp=time_data_tmp+exp_data.time_data;
    end
    test_options.tx_ch=test_options.tx_ch(no_buffs+1:end,:);
    test_options.rx_ch=test_options.rx_ch(no_buffs+1:end,:);
    
    test_options.tx_delay_law_array=test_options.tx_delay_law_array(no_buffs+1:end,:);
    test_options.rx_delay_law_array=test_options.rx_delay_law_array(no_buffs+1:end,:);
    
    exp_data.time_data=time_data_tmp/software_averages;
    
    exp_data.time_data=exp_data.time_data(:, (no_buffs*exp_data.num_els +1):end);%removing buffer captures
    exp_data.time=[1:test_options.time_pts]/test_options.sample_freq;
    
    if save_data
        eval(['save focussed_slice' int2str(nn) ' exp_data'])
    end
    
    disp(['Max value in focussed capture: ' num2str(max(max(exp_data.time_data)))])
    
    for mm=1:Nt
        
        [loc1 loc2]=ind2sub(size(mesh.x),(nn-1)*N+mm);
        
        %Parallel energy calulation
        clear current_capture
        current_capture=exp_data.time_data(:,((mm-1)*exp_data.num_els+1):mm*exp_data.num_els);
        maxval=max(max(current_capture));
        current_capture(1:max_offset_el,:)=0;%remove delayed elements
        current_capture=fft(current_capture);
        current_capture=current_capture(lower_el:upper_el,:);
        image_full(loc1,loc2)=sqrt(trapz(trapz(freq_mat_window_sq.*abs(current_capture).^2)));  %parallel energy calulation
       
        %Sequential energy calculation
        delay_law_array=focal_law.delay_law_vec(((nn-1)*N+mm),:);
        delay_law_array=round(delay_law_array/1e-9)*1e-9;
        use_amps=focal_law.amps(((nn-1)*N+mm),:);
        focused_mat=focused_mat*0;
        
        for kk=1:exp_data.num_els %loop through transmissions
            single_set=squeeze(full_matrix_data(:,:,kk)).*use_amps(kk);
            freq_delay=exp(-2*pi*j*delay_law_array(kk)*freq_arr);
            focused_mat=focused_mat+single_set.*(freq_delay*gpuones);
        end
        
        focused_mat=(ifft(focused_mat));
        focused_mat(1:max_offset_el,:)=0;%remove delayed elements
        focused_mat=fft(focused_mat);
        
        image_FMC(loc1,loc2)=gather(sqrt(trapz(trapz(freq_mat_window_sq.*abs(focused_mat(lower_el:upper_el,:)).^2)))); %sequential energy calculation      
        nonlinear_image_data(loc1,loc2)=gather(-((image_full(loc1,loc2)-image_FMC(loc1,loc2))./(image_full(loc1,loc2))));
       
    end
    
    if nn==1
    figure(1);
    cmap_tmp=colormap('hot');cmap_tmp=flipud(cmap_tmp);
    plothan=imagesc(data.x,data.z,nonlinear_image_data);title('Nonlinear Image');colorbar;axis equal;axis tight;colormap(cmap_tmp);%caxis([0 0.03]);
    else
       set(plothan,'CData', nonlinear_image_data);
       drawnow
    end
    disp(['Focused capture: ' num2str(nn/(ceil(numel(mesh.x)/N))*100)  '%'])
end
if save_data
    save all_nl
end

nonlinear_image_data=gather(nonlinear_image_data);

figure;imagesc(data.x,data.z,nonlinear_image_data);title('Nonlinear Image');colorbar;axis equal;axis tight;colormap(cmap_tmp);

fclose(tcpip_obj)
delete(tcpip_obj)
clear tcpip_obj