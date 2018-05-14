function fn_set_encoder_test_options(encoder_options, varargin)

if nargin > 1
    time_out = varargin{1};
else
    time_out = 10;
end
if nargin > 2
    echo_on = varargin{2};
else
    echo_on = 0;
end

MPE = encoder_options.MPE;
TERM =encoder_options.termination;
BKL = encoder_options.backlash;
SPA = encoder_options.SPA;


% setup the encoder
fn_ag_send_command(sprintf('MPE %i %i %i %i', MPE,MPE,MPE,MPE), 0, echo_on);
fn_ag_send_command(sprintf('TERM %i %i', TERM, TERM), 0, echo_on);
fn_ag_send_command(sprintf('BKL %i %i %i %i', BKL,BKL,BKL,BKL), 0, echo_on);
fn_ag_send_command(sprintf('SPA %i %i %i %i', SPA,SPA,SPA,SPA), 0, echo_on);


return;