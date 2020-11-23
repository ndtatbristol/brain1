%Extract subset of SHM data from bristol tank into a single file

input_folder = 'X:\Projects\2012 - on SHM on tank data';
input_name_template = 'tank-%i-%i-%i-%i-%i.mat'; %yr, mnth, day, hour, min

output_folder = 'C:\Users\mepdw\OneDrive - University of Bristol\Projects\Current projects\2018-20 Turing Fellowship\My NN\SHM idea';

%output file will contain info based on following filters
start_date = datetime(2013,01,01,0,0,0); %year, month, day, hour, min, secs
end_date = datetime(2013,12,31,0,0,0); %year, month, day, hour, min, secs
frequency = 'daily'; %options are 'all', 'monthly', 'weekly', 'daily', 'hourly'
tx_sensors = [1];
rx_sensors = [8];

%--------------------------------------------------------------------------

tmp = dir(fullfile(input_folder, strrep(input_name_template, '%i','*')));

fnames = {tmp(:).name}';

%convert each filename to a date
datetimes = datetime('now'): datetime('now') + length(fnames) - 1;
for ii = 1:length(fnames)
    datetimes(ii) = datetime([sscanf(fnames{ii}, input_name_template); 0]');
end
[datetimes, fi] = sort(datetimes);

%work out which files are in date range
use_file = zeros(length(fnames), 1);
ii = 1;
min_date = start_date;

while ii <= length(fnames)
    current_datetime = datetimes(ii);
    if current_datetime < min_date
        ii = ii + 1;
        continue;
    end
    if current_datetime > end_date
        break;
    end
    use_file(ii) = 1;
    [y,m,d] = ymd(current_datetime);
    [h,mins,s] = hms(current_datetime);
    switch frequency
        case 'all'
        case 'hourly'
            min_date = current_datetime + 1/24;
        case 'daily'
            min_date = current_datetime + 1;
        case 'weekly'
            min_date = current_datetime + 7;
        case 'monthly'
            min_date = datetime(y, m+1, d, h, mins, s);
        case 'quarterly'
            min_date = datetime(y, m+3, d, h, mins, s);
        case 'yearly'
            min_date = datetime(y+1, m, d, h, mins, s);
    end
    ii = ii + 1;
end
file_indices = fi(logical(use_file));

for ii = 1:length(file_indices)
    load(fullfile(input_folder, fnames{file_indices(ii)}));
    if ii == 1
        data_out.time = data1.time;
        jj = find(ismember(data1.tx, tx_sensors) & ismember(data1.rx, rx_sensors));
        data_out.time_data = zeros(length(data_out.time), length(jj), length(file_indices));
        data_out.cycles = data1.cycles;
        data_out.centre_freq = data1.centre_freq;
        data_out.ph_velocity = data1.ph_velocity;
        data_out.tx = data1.tx(jj);
        data_out.rx = data1.rx(jj);
    end
    data_out.time_data(:,:,ii) = data1.time_data(:, jj);
    fn_show_progress(ii, length(file_indices));
end

output_fname = [ datestr(start_date, 'ddmmyyyy'), '-',  datestr(end_date, 'ddmmyyyy'), ' ', frequency, ', Tx[', int2str(tx_sensors),'] Rx[', int2str(rx_sensors),'].mat'];
save(fullfile(output_folder, output_fname), 'data_out');