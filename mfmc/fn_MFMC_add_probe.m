function PROBE = fn_MFMC_add_probe(MFMC, PROBE)
%SUMMMAY
%   Adds probe data to file.
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   PROBE - probe data to add with mandatory fields:
%       .CENTRE_FREQUENCY
%       .ELEMENT_POSITION
%       .ELEMENT_MAJOR
%       .ELEMENT_MINOR
%       .ELEMENT_SHAPE
%       and other optional fields as per the MFMC file specification
%OUTPUTS
%   PROBE - updated copy of PROBE with additional fields:
%       .TYPE = 'PROBE'
%       .name - name of probe group in file relative to MFMC.root_path
%       .ref - HDF5 reference to probe group in file
%       .location - complete path to probe group in file
%--------------------------------------------------------------------------

%generate next free name if name not specified
if ~isfield(PROBE, 'name')
    [PROBE.name,  PROBE.index] = fn_hdf5_next_group_name(MFMC.fname, MFMC.root_path, MFMC.probe_name_template);
end

%create the probe group
probe_path = [MFMC.root_path, PROBE.name, '/'];
PROBE.location = probe_path;

%get the hdf5 reference for the probe group, needed for subsequent focal laws
PROBE.ref = fn_hdf5_create_group(MFMC.fname, probe_path);

%all probes must have TYPE attribute set to 'PROBE'
PROBE.TYPE = 'PROBE';

%mandatory attributes
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'TYPE'],                 'M', 'A');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'CENTRE_FREQUENCY'],     'M', 'A');

%mandatory datasets
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'ELEMENT_POSITION'],     'M', 'D');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'ELEMENT_MAJOR'],        'M', 'D');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'ELEMENT_MINOR'],        'M', 'D');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'ELEMENT_SHAPE'],        'M', 'D');

%optional attributes
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'WEDGE_SURFACE_POINT'],  'O', 'A');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'WEDGE_SURFACE_NORMAL'], 'O', 'A');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'BANDWIDTH'],            'O', 'A');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'PROBE_MANUFACTURER'],   'O', 'A');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'PROBE_SERIAL_NUMBER'],  'O', 'A');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'PROBE_TAG'],            'O', 'A');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'WEDGE_MANUFACTURER'],   'O', 'A');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'WEDGE_SERIAL_NUMBER'],  'O', 'A');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'WEDGE_TAG'],            'O', 'A');

%optional datasets
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'ELEMENT_RADIUS_OF_CURVATURE'],  'O', 'D');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'ELEMENT_AXIS_OF_CURVATURE'],    'O', 'D');
fn_hdf5_create_entry(PROBE, MFMC.fname, [probe_path, 'DEAD_ELEMENT'],         'O', 'D');

end
