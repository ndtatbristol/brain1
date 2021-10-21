function [res, groups] = fn_hdf5_read_to_matlab(fname, location, varargin)
%SUMMARY
%   Reads all datasets and attributes at location into fields of same name
%   in res. Groups are returned as list, but not examined. Optional
%   argument lists names of any datasets to exclude

included_only=0;
if length(varargin) < 1
    datasets_to_treat_specially = [];
else
    if (length(varargin) < 2 || varargin{2} < 1)
        datasets_to_treat_specially = varargin{1};
    else
        datasets_to_treat_specially = varargin{1};
        included_only=1;
    end
end

a = h5info(fname, location);
for ii = 1:length(a.Datasets)
    nm = a.Datasets(ii).Name;
    skip = included_only-0;
    for jj = 1:length(datasets_to_treat_specially)
        if strcmp(nm, datasets_to_treat_specially{jj})
            skip = 1-included_only;
            break;
        end
    end
    if skip
        continue
    end
    if strcmp(a.Datasets(ii).Datatype.Class, 'H5T_REFERENCE')
        %special case of references
        fid = H5F.open(fname);
        dataset_id = H5D.open(fid,[location, '/', nm]);
        res.(nm) = H5D.read(dataset_id)';
        H5F.close(fid);
        H5D.close(dataset_id);
    else
        res.(nm) = h5read(fname, [location, '/', nm]);
    end
end
if (included_only < 1)
    for ii = 1:length(a.Attributes)
        nm = a.Attributes(ii).Name;
        res.(nm) = h5readatt(fname, location, nm);
    end
else
    for ii = 1:length(a.Attributes)
        nm = a.Attributes(ii).Name;
        if strcmp(nm, 'TYPE')
            res.(nm) = h5readatt(fname, location, nm);
            break;
        end
    end
end

if (~exist('res','var'))
    % No datasets found. Return empty structure
    res=[];
end

groups = a.Groups;

end