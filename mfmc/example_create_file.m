%SUMMARY
%   This is a simple example of how to create a new HDF5 file containing an
%   MFMC data structure from Matlab. The numerical values used in the file
%   are random numbers.

%Clear everything and restore path to default
clear;
close all;
clc;
restoredefaultpath;

%Name of MFMC file to create
fname = 'example mfmc file.mfmc';

%--------------------------------------------------------------------------
%CREATING FILE

%If file already exists, delete it
if exist(fname, 'file')
    delete(fname);
end

%Create MFMC file and MFMC structure needed for subsequent functions (this
%MFMC variable should not be altered)
MFMC = fn_MFMC_open_file(fname);

%--------------------------------------------------------------------------
%CREATING A PROBE

%Define 1D probe with following physical parameters
el_number = 8; 
el_pitch = 1e-3; 
el_length = 10e-3; 
el_width = 0.8e-3; 
cent_freq = 5e6;

%Populate a Matlab data structure with the required fields that match 
%mandatory fields in MFMC structure specification
PROBE.CENTRE_FREQUENCY = cent_freq;
el_x_pos = ([1:el_number] - (el_number + 1)/2) * el_pitch;                                              %element x coordinates in PCS are equispaced at appropriate pitch with centre at x = 0.
PROBE.ELEMENT_POSITION = [el_x_pos; zeros(2, el_number)];         
PROBE.ELEMENT_MAJOR = [zeros(1, el_number); ones(1, el_number) * el_length / 2; zeros(1, el_number)];   %major vectors go from element centre to mid-points of short side
PROBE.ELEMENT_MINOR = [ones(1, el_number) * el_width / 2; zeros(2, el_number)];                         %minor vectors go from element centre to mid-points of long side
PROBE.ELEMENT_SHAPE = ones(1, el_number);                                                               %ELEMENT_SHAPE = 1 is rectangular

%Add probe details to MFMC file
PROBE = fn_MFMC_add_probe(MFMC, PROBE);                                                                 %PROBE returned has extra fields, including the HDF5 object reference (PROBE.ref) and location in file (PROBE.location)

%Example of adding user attribute to probe in file once probe has been added
fn_MFMC_add_user_attribute(MFMC, PROBE.location, 'USER_ATTRIBUTE', 'Special information');

%--------------------------------------------------------------------------
%CREATING A SEQUENCE

%Define a sequence with following physical parameters
sample_freq = 100e6;
start_time = 0;
long_vel = 6e3;
shear_vel = 3e3;

%Populate a Matlab data structure with the required fields that match 
%mandatory fields in MFMC structure specification
SEQUENCE.TIME_STEP = 1/sample_freq;
SEQUENCE.START_TIME = start_time;
SEQUENCE.SPECIMEN_VELOCITY = [shear_vel, long_vel];
SEQUENCE.PROBE_LIST = PROBE.ref; %this is the HDF5 object reference to the probe created above

%Example focal laws here are for plane wave transmission at 10degrees 
%(demonstrating multiple elements active) and reception on individual
%elements. There is no distinction between transmission and reception when
%focal laws are defined, hence need el_number + 1 laws in total; first
%el_number are the individual elements ones that will be used for
%receptions and el_number+1 law will be used for transmission
plane_wave_angle_degrees = 10;

for jj = 1:el_number
    SEQUENCE.LAW{jj}.ELEMENT = int32(jj);                                                        %identify individual element
    SEQUENCE.LAW{jj}.PROBE = PROBE.ref;                                                                 %reference to probe
end
SEQUENCE.LAW{el_number+1}.ELEMENT = int32([1:el_number]);                                               %identify all elements
SEQUENCE.LAW{el_number+1}.PROBE = repmat(PROBE.ref, [el_number, 1]);                                    %need reference to probe for each active element
SEQUENCE.LAW{el_number+1}.DELAY = el_x_pos * sind(plane_wave_angle_degrees) / long_vel;      %delays for plane wave at angle

%Now define the focal laws for transmission and reception associated with
%each A-scan in data. In Matlab these are defined by indices referring to
%the focal laws above; in the file, these are converted into HDF5 object 
%references
SEQUENCE.transmit_law_index = ones(1, el_number) * (el_number + 1);                                     %transmit focal law is same for all A-scans
SEQUENCE.receive_law_index = [1: el_number];                                                            %different receive focal law for each A-scan

%Add the sequence details to the file
SEQUENCE = fn_MFMC_add_sequence(MFMC, SEQUENCE);                                                        %SEQUENCE returned has extra fields, including the HDF5 object reference (SEQUENCE.ref) and location in file (SEQUENCE.location)

%Example of adding user dataset to sequence in file once sequence has been added
fn_MFMC_add_user_dataset(MFMC, SEQUENCE.location, 'USER_DATASET', rand(10,10));
%--------------------------------------------------------------------------
%ADD TWO FRAMES OF DATA TO SEQUENCE

no_a_scans_per_frame = el_number;
no_time_pts = 100000;

FRAME.MFMC_DATA = int8(randi(2^8, no_time_pts, no_a_scans_per_frame)-2^7);                              %random int8 data to represent physical data
FRAME.PROBE_POSITION = [0; 0; 0];                                                                       %PCS origin is at GCS origin
FRAME.PROBE_X_DIRECTION = [1; 0; 0];                                                                    %PCS x-axis is aligned to GCS x-axis
FRAME.PROBE_Y_DIRECTION = [0; 1; 0];                                                                    %PCS y-axis is aligned to GCS y-axis
FRAME.PROBE_PLACEMENT_INDEX = ones(no_a_scans_per_frame, 1);                                            %refer to first probe position for all A-scans in frame
% FRAME.deflate_value = 0; %uncomment to set a compression value from 0
% (none) to 9 (max), otherwise default of 4 is used. Does not make any
% difference when data is completely random integers stored as int8 anyway.

%Add the frame to the previous sequence
fn_MFMC_add_frame(MFMC, SEQUENCE.ref, FRAME);

%Generate a second frame of data when the probe is translated by 3mm in x
%direction of GCS (everything else remains same)
FRAME.MFMC_DATA = double(randi(2^8, no_time_pts, no_a_scans_per_frame)-2^7);                              %some new random int8 data to represent physical data
FRAME.PROBE_POSITION = [3e-3; 0; 0];
FRAME.PROBE_PLACEMENT_INDEX = ones(no_a_scans_per_frame, 1) * 2;                                        %refer to second probe position for all A-scans in frame

%Add the frame to the previous sequence
fn_MFMC_add_frame(MFMC, SEQUENCE.ref, FRAME);

%--------------------------------------------------------------------------
%DISPLAY FILE SUMMARY IN COMMAND WINDOW
h5disp(fname);
tmp = dir(fname);
fprintf('File size: %.2f MB\n', tmp.bytes / 1e6);