function [PROBE]=fn_MFMC_helper_brain_array_to_probe(array,varargin)

    % Convert BRAIN array data structure to PROBE of MFMC format

    %Populate a Matlab data structure with the required fields that match 
    %mandatory fields in MFMC structure specification
    PROBE.CENTRE_FREQUENCY = array.centre_freq;
    
    PROBE.ELEMENT_POSITION = [array.el_xc(:), array.el_yc(:), array.el_zc(:)]';
    
    dx = max(abs(array.el_xc-array.el_x1),abs(array.el_xc-array.el_x2))';
    dy = max(abs(array.el_yc-array.el_y1),abs(array.el_yc-array.el_y2))';
    %dz=2*max(abs(array.el_zc-array.el_z1),abs(array.el_zc-array.el_z2));
    zz = zeros(size(dy));
    PROBE.ELEMENT_MAJOR = [zz; dy; zz];
    PROBE.ELEMENT_MINOR = [dx; zz; zz];
    
    if (~isempty(varargin))
        switch varargin{1} %shape type
            case 'rectangular'
                PROBE.ELEMENT_SHAPE = ones(1, length(array.el_xc));  %1 is rectangular
            case 'elliptical'
                PROBE.ELEMENT_SHAPE = 2*ones(1, length(array.el_xc));  %2 is elliptical
        end
    else
        % Assumed rectangular
        PROBE.ELEMENT_SHAPE = ones(1, length(array.el_xc));  %1 is rectangular
    end
    
    
    
end