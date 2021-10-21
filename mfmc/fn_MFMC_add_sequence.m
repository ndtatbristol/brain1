function SEQUENCE = fn_MFMC_add_sequence(MFMC, SEQUENCE)
%SUMMMAY
%   Adds sequence data with optional frames to file.
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   SEQUENCE - sequence data to add with mandatory fields:
%       .transmit_law_index - indices of transmit focal law associated with 
%       each A-scan in frame of MFMC data
%       .receive_law_index - indices of receive focal law associated with 
%       each A-scan in frame of MFMC data
%       .LAW{index} - cell array of focal laws with mandatory fields
%           .PROBE - HDF5 reference to probe(s) used in focal law
%           .ELEMENT - index of element(s) used in focal law
%       .TIME_STEP
%       .START_TIME
%       .SPECIMEN_VELOCITY
%       .PROBE_LIST
%       and other optional fields as per the MFMC file specification
%OUTPUTS
%   SEQUENCE - updated copy of SEQUENCE with additional fields:
%       .TYPE = 'SEQUENCE'
%       .TRANSMIT_LAW - HDF5 references to transmit law groups in file
%       .RECEIVE_LAW - HDF5 references to receive law groups in file
%       .name - name of sequence group in file relative to MFMC.root_path
%       .ref - HDF5 reference to sequence group in file
%       .location - complete path to sequence group in file
%--------------------------------------------------------------------------

% if length(varargin) < 1
%     deflate_level = 5;
% else
%     deflate_level = varargin{1};
% end

if ~isfield(SEQUENCE, 'name')
    [SEQUENCE.name,  SEQUENCE.index] = fn_hdf5_next_group_name(MFMC.fname, MFMC.root_path, MFMC.sequence_name_template);
end

sequence_path = [MFMC.root_path, SEQUENCE.name, '/'];
SEQUENCE.ref = fn_hdf5_create_group(MFMC.fname, sequence_path);
SEQUENCE.location = sequence_path;
SEQUENCE.TYPE = 'SEQUENCE';

%sort out the focal laws as they need to be referenced later
for ii = 1:length(SEQUENCE.LAW)
    if ~isfield(SEQUENCE.LAW{ii}, 'name')
        SEQUENCE.LAW{ii}.name = sprintf(MFMC.law_name_template, ii);
    end
    law_path = [sequence_path, SEQUENCE.LAW{ii}.name, '/'];
    SEQUENCE.LAW{ii}.location = law_path;
    
    %get the hdf5 reference for the law group, needed for cross-referencing
    SEQUENCE.LAW{ii}.ref = fn_hdf5_create_group(MFMC.fname, law_path);
    SEQUENCE.LAW{ii}.TYPE = 'LAW';
    
    %mandatory attributes
    fn_hdf5_create_entry(SEQUENCE.LAW{ii}, MFMC.fname, [law_path, 'TYPE'],          'M', 'A');
    
    %mandatory datasets
    fn_hdf5_create_entry(SEQUENCE.LAW{ii}, MFMC.fname, [law_path, 'PROBE'],         'M', 'D', 'hdf5_ref');
    fn_hdf5_create_entry(SEQUENCE.LAW{ii}, MFMC.fname, [law_path, 'ELEMENT'],       'M', 'D', 'int32');
    
    %optional datasets
    fn_hdf5_create_entry(SEQUENCE.LAW{ii}, MFMC.fname, [law_path, 'DELAY'],         'O', 'D');
    fn_hdf5_create_entry(SEQUENCE.LAW{ii}, MFMC.fname, [law_path, 'WEIGHTING'],     'O', 'D');
end

%mandatory attributes
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'TYPE'],                 'M', 'A');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'TIME_STEP'],            'M', 'A');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'START_TIME'],           'M', 'A');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'SPECIMEN_VELOCITY'],    'M', 'A');

%mandatory datasets
for ii = 1:numel(SEQUENCE.transmit_law_index)
    SEQUENCE.TRANSMIT_LAW(ii,:) = SEQUENCE.LAW{SEQUENCE.transmit_law_index(ii)}.ref;
end
for ii = 1:numel(SEQUENCE.receive_law_index)
    SEQUENCE.RECEIVE_LAW(ii,:) = SEQUENCE.LAW{SEQUENCE.receive_law_index(ii)}.ref;
end
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'TRANSMIT_LAW'],         'M', 'D', 'hdf5_ref');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'RECEIVE_LAW'],          'M', 'D', 'hdf5_ref');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'PROBE_LIST'],                'M', 'D', 'hdf5_ref');%TODO

%optional attributes
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'WEDGE_VELOCITY'],       'O', 'A');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'TAG'],                  'O', 'A');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'RECEIVER_AMPLIFIER_GAIN'], 'O', 'A');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'FILTER_TYPE'],          'O', 'A', 'int32');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'FILTER_PARAMETERS'],    'O', 'A');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'FILTER_DESCRIPTION'],   'O', 'A');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'OPERATOR'],             'O', 'A');
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'DATE_AND_TIME'],        'O', 'A');

%optional datasets
fn_hdf5_create_entry(SEQUENCE, MFMC.fname, [sequence_path, 'DAC_CURVE'],            'O', 'D');

%add MFMC if any present at this point (can be added later anyway)
if isfield(SEQUENCE, 'FRAME')
    fn_MFMC_add_frame(MFMC, SEQUENCE.ref, SEQUENCE.FRAME)
end
end

