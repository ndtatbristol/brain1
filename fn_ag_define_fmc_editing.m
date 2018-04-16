function fn_ag_define_fmc_editing(transmitters, receivers, hmc, echo_on)
%transmitters is matrix where each row is a list of transmitters to be used
%in each transmission cycle - for FMC this is just a column vector [1:n]'
%receivers is a single vector of receivers to be used in each transmit cycle
%set up focal laws

tx_matrix = diag(ones(length(transmitters),1))
rx_matrix = zeros(size(tx_matrix));
rx_matrix(1:end,receivers) = 1

if hmc == 1
    rx_matrix = triu(rx_matrix)
end

for fl_ii = 1:length(transmitters) %loop through focal laws
        
    %clear existing tx delays
    for tx_ii = 1:64 %loop through focal laws
%         fn_ag_send_command(sprintf('TXF %i %i -1', fl_ii, tx_ii), 0, echo_on);%law, ch, del
          sprintf('TXF %i %i -1', fl_ii, tx_ii)
    end;
    %find transmitters for each focal law (i.e each row of the tx or rx_matrix
    tx_nos = find(tx_matrix(fl_ii,:))
    for tx_ii = 1:length(tx_nos) %add each transmitter specified for focal law
%         fn_ag_send_command(sprintf('TXF %i %i 0', fl_ii, transmitters(fl_ii, tx_ii)), 0, echo_on);%law, ch, del
          sprintf('TXF %i %i 0', fl_ii, tx_ii)
    end;
    %clear existing rx delays
%     fn_ag_send_command(sprintf('RXF %i 0 -1 0', fl_ii), 0, echo_on);%law, ch, del
    rx_nos = find(rx_matrix(fl_ii,:))
    for rx_ii = 1:length(rx_nos); %add receivers to all focal laws
%         fn_ag_send_command(sprintf('RXF %i %i 0 0', fl_ii, receivers(rx_ii)), 0, echo_on);%law, ch, del, trim_amp
          sprintf('RXF %i %i 0 0', fl_ii,receivers(rx_ii))
    end;
    %assign focal laws to tests starting at 256
%     fn_ag_send_command(sprintf('TXN %i %i', 255 + fl_ii, fl_ii), 0, echo_on);
%     fn_ag_send_command(sprintf('RXN %i %i', 255 + fl_ii, fl_ii), 0, echo_on);
    sprintf('TXN %i %i', 255 + fl_ii, fl_ii)
    sprintf('RXN %i %i', 255 + fl_ii, fl_ii)
end;

%assign tests to sweep 1
if size(transmitters, 1) > 1
    fn_ag_send_command(sprintf('SWP 1 %i - %i',256 ,255 + size(transmitters, 1)), 0, echo_on);
else
    fn_ag_send_command(sprintf('SWP 1 %i',256), 0, echo_on);
end;
%disable all sweeps
fn_ag_send_command('DIS 0', 0, echo_on);
fn_ag_send_command('DISS 0', 0, echo_on);

%enable test 256
fn_ag_send_command('ENA 256', 0, echo_on);
%enable sweep 1
fn_ag_send_command('ENAS 1', 0, echo_on);
return;
