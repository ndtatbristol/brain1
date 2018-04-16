function fn_timer_start(varargin);
%USAGE
%   fn_timer_start([str, show_result])
%SUMMARY
%   Wrapper for Matlab tic function, with optional text output that can
%   be enabled for debugging or disabled for speed. Use with fn_timer_end
%AUTHOR
%	Paul Wilcox (Oct 2007)
%INPUTS
%   str[''] - string to display (e.g. 'Process xxx started');
%   show_result[1] - enable or disable string display
%OUTPUTS
%   none

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%set default values
if nargin < 1
    str = '';
else
    str = varargin{1};
end
if nargin < 2
    show_result = 1;
else
    show_result = varargin{2};
end;
%show string if appropriate
if show_result & ~isempty(str)
    disp(str);
end;
%start timer
tic;
return;