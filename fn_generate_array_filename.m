function fname = fn_generate_array_filename(array)
%SUMMARY
%   Generates array filename (including .mat) from array structure. Handles
%   regular 1D linear arrays, equally split regular 1D arrays, 2D matrix
%   arrays. 2D non-matrix arrays will be identified as random and the
%   minimum pitch given.
%INPUTS
%   array - structured variable describing array containing fields: 'el_xc',
%   'el_yc' and optionally 'manufacturer' and 'centre_freq'
%OUTPUTS
%   fname - generated filename
%--------------------------------------------------------------------------

if isfield(array, 'manufacturer')
    manufacturer = [array.manufacturer, ' '];
else
    manufacturer = '';
end
if isfield(array, 'centre_freq')
    centre_freq_str = sprintf('%.2fMHz ', array.centre_freq /1e6);
else
    centre_freq_str = '';
end

els = length(array.el_xc);


%find minimum separation between elements
[ii, jj] = meshgrid(1:els, 1:els);
d = sqrt((array.el_xc(ii) - array.el_xc(jj)) .^ 2 + (array.el_yc(ii) - array.el_yc(jj)) .^ 2) + eye(els) * 1e10;
pitch = min(min(d));

%decide if 1D or 2D
if any(array.el_yc)
    type = '2D ';
    if length(unique(array.el_xc)) * length(unique(array.el_yc)) == els
        type = [type, 'matrix '];
        pitch_str = sprintf('%.2fmm pitch', pitch * 1e3);
    else
        type = [type, 'random '];
        pitch_str = sprintf('%.2fmm min pitch', pitch * 1e3);
    end
    els_string = sprintf('%iels ', els);
else
    type = '1D ';
    %decide if twin array
    if length(unique(array.el_xc)) < els
        els_string = sprintf('2x%iels ', round(els / 2));
        pitch = abs(array.el_xc(2) - array.el_xc(1)); %pitch has to be calculated differently in this case
    else
        els_string = sprintf('%iels ', els);
    end
    pitch_str = sprintf('%.2fmm pitch', pitch * 1e3);
end;

fname = [manufacturer, type, els_string, centre_freq_str, pitch_str, '.mat'];
end