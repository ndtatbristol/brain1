function [SEQUENCE_candidate]=fn_MFMC_helper_add_sequence_if_new(MFMC,SEQUENCE_candidate)

% Subroutine looks within MFMC file and adds sequence if not already present
% Returned value is either matched sequence already in file, or updated new
% sequence (with ref,location,index etc added)
    
    [~,current_sequence_list] = fn_MFMC_get_probe_and_sequence_refs(MFMC);

    for i=1:length(current_sequence_list)
        SEQUENCE = fn_MFMC_read_sequence(MFMC, current_sequence_list{i}.ref);
        
        % Need to compare what SEQUENCE_candidate would become in HDF5
        % format with SEQUENCE already in HDF5 format
        
        % Compare candidate to this sequence (mandatory fields only)
        if (length(SEQUENCE.PROBE_LIST) ~= length(SEQUENCE_candidate.PROBE_LIST))
            continue;
        end
        if (max(SEQUENCE.PROBE_LIST~=SEQUENCE_candidate.PROBE_LIST))
            continue;
        end
        if (max(abs(SEQUENCE.SPECIMEN_VELOCITY(:)-SEQUENCE_candidate.SPECIMEN_VELOCITY(:)))>1e-7)
            continue;
        end
        if (isfield(SEQUENCE_candidate,'WEDGE_VELOCITY'))
            if (~isfield(SEQUENCE,'WEDGE_VELOCITY'))
                continue;
            end
            if (max(isnan(SEQUENCE.WEDGE_VELOCITY(:)) ~= isnan(SEQUENCE_candidate.WEDGE_VELOCITY(:))))
                continue;
            end
            if (max(abs(SEQUENCE.WEDGE_VELOCITY(~isnan(SEQUENCE.WEDGE_VELOCITY))-SEQUENCE_candidate.WEDGE_VELOCITY(~isnan(SEQUENCE_candidate.WEDGE_VELOCITY))))>1e-7)
                continue;
            end
        end
        if (abs(SEQUENCE.TIME_STEP-SEQUENCE_candidate.TIME_STEP)>1e-15)
            continue;
        end
        if (abs(SEQUENCE.START_TIME-SEQUENCE_candidate.START_TIME)>1e-15)
            continue;
        end
        if (size(SEQUENCE.TRANSMIT_LAW,1) ~= length(SEQUENCE_candidate.transmit_law_index))
            continue;
        end
        %check each law in turn (but only check each TX/RX law once for speed)
        trans_tmp=uint8(zeros(8,length(SEQUENCE_candidate.LAW)));
        trans_tmp2=zeros(1,length(SEQUENCE_candidate.LAW));
        rec_tmp=uint8(zeros(8,length(SEQUENCE_candidate.LAW)));
        rec_tmp2=zeros(1,length(SEQUENCE_candidate.LAW));
        transmit=SEQUENCE.TRANSMIT_LAW.';
        receive=SEQUENCE.RECEIVE_LAW.';
        match=1;
        for ii=1:length(SEQUENCE_candidate.transmit_law_index)
            itx=SEQUENCE_candidate.transmit_law_index(ii);
            irx=SEQUENCE_candidate.receive_law_index(ii);
            %TX law
            ref_tx=transmit(:,ii).';
            if (trans_tmp2(itx) > 0)
                % Previously loaded this law, just check law number matches
                if (max(trans_tmp(:,itx)-ref_tx) > 0)
                   match=0;
                   disp(['Law (TX) does not match ',num2str(ii),' ',num2str(itx)])
                   break;
                end
            else
                 law = fn_MFMC_read_law(MFMC,ref_tx );
                 trans_tmp(:,itx)=ref_tx;
                 trans_tmp2(itx)=1;
                 if (SEQUENCE_candidate.LAW{itx}.ELEMENT ~= law.ELEMENT || max(SEQUENCE_candidate.LAW{itx}.PROBE ~= law.PROBE))
                    match=0;
                    disp(['Law (TX) does not match ',num2str(ii),' ',num2str(itx)])
                    break;
                 end
            end
            %RX law
            ref_rx=receive(:,ii).'; 
            if (rec_tmp2(irx) > 0)
                % Previously loaded this law, just check law number matches
                if (max(rec_tmp(:,irx)-ref_rx) > 0)
                   match=0;
                   disp(['Law (RX) does not match ',num2str(ii),' ',num2str(irx)])
                   break;
                end
            else
                 law = fn_MFMC_read_law(MFMC,ref_rx );
                 rec_tmp(:,irx)=ref_rx;
                 rec_tmp2(irx)=1;
                 if (SEQUENCE_candidate.LAW{irx}.ELEMENT ~= law.ELEMENT || max(SEQUENCE_candidate.LAW{irx}.PROBE ~= law.PROBE))
                    match=0;
                    disp(['Law (RX) does not match ',num2str(ii),' ',num2str(irx)])
                    break;
                 end
            end        
        end
        if (match < 1)
            continue;
        end
        SEQUENCE.ref = current_sequence_list{i}.ref;
        SEQUENCE.location = current_sequence_list{i}.location;
        SEQUENCE.name = current_sequence_list{i}.name;
        SEQUENCE.index = i;
        disp(['Sequence matched at index ',num2str(i)])
        SEQUENCE_candidate=SEQUENCE;
        return;
    end
    
    %No match found, since got to this line
    disp('Adding sequence to file')
    %Add sequence details to MFMC file
    SEQUENCE_candidate = fn_MFMC_add_sequence(MFMC, SEQUENCE_candidate);  
    
end
    
   