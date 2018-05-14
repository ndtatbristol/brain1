%set up focal laws
function [tx_no, rx_no] = fn_DSL_define_fmc(tx_ch, rx_ch, echo_on)

transmit_laws = size(tx_ch, 1);
time_traces = length(find(rx_ch));
tx_no=zeros(time_traces,1)';
rx_no=zeros(time_traces,1)';
counter = 0;

for fl_ii = 1:transmit_laws %loop through focal laws

    %clear existing tx delays
%         for tx_ii = 1:transmit_laws %loop through focal laws
%             fn_ag_send_command(sprintf('TXF %i %i -1', fl_ii, tx_ii), 0, echo_on);%law, ch, del
%         end;
    %find transmitters for each focal law (i.e each row of the tx or rx_matrix
    tx_nos = find(tx_ch(fl_ii,:));
%         for tx_ii = 1:length(tx_nos) %add each transmitter specified for focal law
%             fn_ag_send_command(sprintf('TXF %i %i 0', fl_ii, tx_nos(tx_ii)), 0, echo_on);%law, ch, del
%         end;
    %clear existing rx delays
%         fn_ag_send_command(sprintf('RXF %i 0 -1 0', fl_ii), 0, echo_on);%law, ch, del
    rx_nos = find(rx_ch(fl_ii,:));
    for rx_ii = 1:length(rx_nos); %add receivers to all focal laws
        counter = counter + 1;
%             fn_ag_send_command(sprintf('RXF %i %i 0 0', fl_ii, rx_nos(rx_ii)), 0, echo_on);%law, ch, del, trim_amp

        if length(tx_nos)>1
            tx_no(counter) = 1;
        else
            tx_no(counter) = tx_nos;
        end

        rx_no(counter) = rx_nos(rx_ii);
    end;
    %assign focal laws to tests starting at 256
%         fn_ag_send_command(sprintf('TXN %i %i', 255 + fl_ii, fl_ii), 0, echo_on);
%         fn_ag_send_command(sprintf('RXN %i %i', 255 + fl_ii, fl_ii), 0, echo_on);
end;
end