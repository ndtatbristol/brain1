function [current_status] = fn_ag_get_axis1_status(MPE, varargin)
%gets axis 1 position
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

encoders_per_mm = 16;

sta_result = fn_ag_send_command('STA', time_out, echo_on);
current_status = double(sta_result(3)) + double(sta_result(4)) * (2^8) + double(sta_result(5)) * (2^16);
current_status = current_status / (encoders_per_mm/MPE);

return;