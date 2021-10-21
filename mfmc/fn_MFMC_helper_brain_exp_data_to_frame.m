function [SEQUENCE]=fn_MFMC_helper_brain_exp_data_to_frame(exp_data,MFMC, SEQUENCE, varargin)

% Convert BRAIN (relevant) exp_data data structure to FRAME of MFMC format
% Add to specified SEQUENCE in MFMC file

% 1st optional input - probe location in GCS as 3-element vector
% [x,y,z]

% 2nd optional input - probe orientation vector as 3x2 matrix where col
% 1 is unit vector in probe x-axis direction and col 2 is unit vector
% in probe y-axis direction

% 3rd optional input is deflate value from 0 (none) to 9 (max). Default
% is 4

default_deflate_value = 4;
default_probe_orientation_vectors = [1, 0; 0, 1; 0, 0]; %aligned to GCS is default

if length(varargin) >= 1 && ~isempty(varargin{1}) && numel(varargin{1}) == 3
    probe_position = varargin{1}(:);
else
    probe_position = [0; 0; 0];
end
if length(varargin) >= 2 && ~isempty(varargin{2}) && all(size(varargin{2}) == [3,2])
    probe_orientation_vectors = varargin{2};
else
    probe_orientation_vectors = default_probe_orientation_vectors;
end
if length(varargin) >= 3 && ~isempty(varargin{3}) && numel(varargin{3}) == 1 && varargin{3} >= 0 && varargin{3} <= 9
    deflate_value = varargin{3};
else
    deflate_value = default_deflate_value;
end

FRAME.MFMC_DATA = exp_data.time_data;

%Following is UoB-specific - deleted PW 22/5/20
%     conversion_factor=0.0125e-3;
%
%     if (length(varargin) > 0)
%         conversion_factor=varargin{1};
%         if (length(varargin)>1)
%             probe_orientation_vectors=varargin{2};
%         end
%     end

FRAME.PROBE_POSITION = probe_position;
FRAME.PROBE_X_DIRECTION = probe_orientation_vectors(:,1);
FRAME.PROBE_Y_DIRECTION = probe_orientation_vectors(:,2);

% Get number of current positional vectors in SEQUENCE
tmp = fn_MFMC_get_data_dimensions(MFMC, SEQUENCE.ref,'PROBE_POSITION');
if (~isempty(tmp))
    counter = tmp(3);
else
    counter = 1;
end

FRAME.PROBE_PLACEMENT_INDEX = ones(length(exp_data.tx), 1) * counter;

FRAME.deflate_value = deflate_value;

%Add the frame to the specified sequence
fn_MFMC_add_frame(MFMC, SEQUENCE.ref, FRAME);

%Following is UoB-specific - deleted PW 22/5/20
% %Add probe standoff and angle if known
% if (isfield(exp_data,'location') && isfield(exp_data.location,'standoff'))
%     %Example of adding user dataset to sequence in file once sequence has been added
%     fn_MFMC_add_user_dataset(MFMC, SEQUENCE.location, 'USER_PROBE_STANDOFF', exp_data.location.standoff);
% end
% if (isfield(exp_data,'location') && isfield(exp_data.location,'angle1'))
%     %Example of adding user dataset to sequence in file once sequence has been added
%     fn_MFMC_add_user_dataset(MFMC, SEQUENCE.location, 'USER_PROBE_ANGLE', exp_data.location.angle1);
% end


end