function [PROBE_candidate]=fn_MFMC_helper_add_probe_if_new(MFMC,PROBE_candidate)

% Subroutine looks within MFMC file and adds probe if not already present
% Returned value is either matched probe already in file, or updated new
% probe (with ref,location,index etc added)
    
    [current_probe_list, ~] = fn_MFMC_get_probe_and_sequence_refs(MFMC);

    for i=1:length(current_probe_list)
        PROBE = fn_MFMC_read_probe(MFMC, current_probe_list{i}.ref);
        
        % Compare candidate to this probe (mandatory fields only)
        if (max(size(PROBE.ELEMENT_POSITION) ~= size(PROBE_candidate.ELEMENT_POSITION)))
            continue;
        end
        if (abs(PROBE.CENTRE_FREQUENCY-PROBE_candidate.CENTRE_FREQUENCY) > 1e-7)
            continue;
        end
        if (max(abs(PROBE.ELEMENT_POSITION(:)-PROBE_candidate.ELEMENT_POSITION(:))) > 1e-7)
            continue;
        end
        if (max(abs(PROBE.ELEMENT_MINOR(:)-PROBE_candidate.ELEMENT_MINOR(:))) > 1e-7)
            continue;
        end
        if (max(abs(PROBE.ELEMENT_MAJOR(:)-PROBE_candidate.ELEMENT_MAJOR(:))) > 1e-7)
            continue;
        end
        if (max(abs(PROBE.ELEMENT_SHAPE(:)-PROBE_candidate.ELEMENT_SHAPE(:))) > 0)
            continue;
        end

        disp(['Probe matched at index ',num2str(i)])
        PROBE.ref = current_probe_list{i}.ref;
        PROBE.location = current_probe_list{i}.location;
        PROBE.name = current_probe_list{i}.name;
        PROBE.index = i;
        PROBE_candidate=PROBE;
        return;
    end
    
    %No match found, since got to this line
    disp('Adding probe to file')
    %Add probe details to MFMC file
    PROBE_candidate = fn_MFMC_add_probe(MFMC, PROBE_candidate);  
    
end
    
   