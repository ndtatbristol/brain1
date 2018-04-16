function timer_end(varargin);
%USAGE
%   fn_timer_end([str, show_result])
%SUMMARY
%   Wrapper for Matlab tic function, with optional text output that can
%   be enabled for debugging or disabled for speed. Use with fn_timer_end
%AUTHOR
%	Paul Wilcox (Oct 2007)
%INPUTS
%   str['Completed in '] - string to display before time which is displayed
%   in seconds (e.g. 'Process xxx competed in ');
%   show_result[1] - enable or disable string display
%OUTPUTS
%   none

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%set default values
if nargin < 1
    str = 'Completed in ';
else
    str = varargin{1};
end;
if nargin < 2
    show_result = 1;
else
    show_result = varargin{2};
end;
%show result
if show_result
    disp([str, sprintf('%.3f seconds', toc)]);
end;
return;