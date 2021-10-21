function [PROBE, SEQUENCE] = fn_MFMC_get_probe_and_sequence_refs(MFMC)
%SUMMARY
%   Returns references, names and locations of all probe and sequence
%   groups in MFMC structure in HDF5 file
PROBE = fn_hdf5_group_refs_by_type(MFMC.fname, MFMC.root_path, 'PROBE');
SEQUENCE = fn_hdf5_group_refs_by_type(MFMC.fname, MFMC.root_path, 'SEQUENCE');
end