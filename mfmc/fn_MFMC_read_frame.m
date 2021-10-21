function FRAME = fn_MFMC_read_frame(MFMC, ref_or_index_or_loc, frame_index)
%SUMMMARY
%   Reads frame(s) of MFMC data from file based on HDF5 reference, 
%   index or location of sequence and index of frame(s).
%INPUTS
%   MFMC - MFMC structure (see fn_MFMC_open_file)
%   ref_or_index_or_loc - HDF5 reference, index or location (relative to
%   MFMC.root_path) of sequence to read from
%   frame_index - index or indices of frames to read
%OUTPUTS
%   FRAME - 2D (or 3D if multiple frames) matrix of frame(s) of MFMC data 
%--------------------------------------------------------------------------

sequence_path = [fn_hdf5_ref_or_index_or_loc_to_loc(ref_or_index_or_loc, MFMC.fname, [MFMC.root_path, MFMC.sequence_name_template]), '/'];

info = h5info(MFMC.fname, [sequence_path, 'MFMC_DATA']);
count = [info.Dataspace.Size(1:2), 1];

FRAME = zeros([info.Dataspace.Size(1:2), length(frame_index)]);
for ii = 1:length(frame_index)
    start = [1,1,frame_index(ii)];
    FRAME(:,:,ii) = h5read(MFMC.fname, [sequence_path, 'MFMC_DATA'], start, count);
    try
        FRAME(:,:,ii) = FRAME(:,:,ii) + 1i * h5read(MFMC.fname, [sequence_path, 'MFMC_DATA_IM', start, count]);
    catch
    end
end

end