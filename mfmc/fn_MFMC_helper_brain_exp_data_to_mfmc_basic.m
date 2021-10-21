function fn_MFMC_helper_brain_exp_data_to_mfmc_basic(exp_data, fname)
%SUMMARY
%   Stores exp_data as single frame in MFMC file containing single
%   sequence. If file already exists, it is overwritten - no checks are
%   made and data is not appended as new sequence. This enables a simple
%   one-to-one translation of exp_data into an MFMC file; the inverse
%   function is exp_data = fn_MFMC_helper_frame_to_brain_exp_data(fname);

MFMC = fn_MFMC_open_file(fname);

%Create probe from BRAIN's exp_data.array
PROBE=fn_MFMC_helper_brain_array_to_probe(exp_data.array);

%Add probe details to MFMC file
PROBE = fn_MFMC_add_probe(MFMC, PROBE);

SEQUENCE.TIME_STEP = exp_data.time(2) - exp_data.time(1);
SEQUENCE.START_TIME = exp_data.time(1);
long_vel=exp_data.ph_velocity;
SEQUENCE.SPECIMEN_VELOCITY = [NaN, long_vel];
SEQUENCE.PROBE_LIST = PROBE.ref; %this is the HDF5 object reference

for jj = 1:length(exp_data.array.el_xc)
    SEQUENCE.LAW{jj}.ELEMENT = int32(jj);       %identify individual element
    SEQUENCE.LAW{jj}.PROBE = PROBE.ref;         %reference to probe
end
%Now define the focal laws for transmission and reception associated with
%each A-scan in data. In Matlab these are defined by indices referring to
%the focal laws above; in the file, these are converted into HDF5 object
%references
SEQUENCE.transmit_law_index = exp_data.tx;
SEQUENCE.receive_law_index = exp_data.rx;

SEQUENCE = fn_MFMC_add_sequence(MFMC, SEQUENCE);

FRAME.MFMC_DATA = exp_data.time_data;
FRAME.PROBE_PLACEMENT_INDEX = ones(size(exp_data.time_data, 2), 1);
FRAME.PROBE_POSITION = zeros(3, 1);
FRAME.PROBE_X_DIRECTION = zeros(3, 1);
FRAME.PROBE_Y_DIRECTION = zeros(3, 1);

fn_MFMC_add_frame(MFMC, SEQUENCE.ref, FRAME);
end
