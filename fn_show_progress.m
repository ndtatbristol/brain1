function fn_show_progress(current_count, total_count, varargin);
%USAGE
%   fn_show_progress(current_count, total_count, [waitbar_text, cmd_window_version, increment, format_string]);
%SUMMARY
%   Shows progress through a loop as a percentage, but only displays at
%   specified percentage incements to save time.
%   Modified to use waitbar as default in 2009
%AUTHOR
%	Paul Wilcox (Oct 2007)
%INPUTS
%   current_count - current value of loop counter
%   total_count - total number of counts in loop
%   waitbar_text['Please wait'] - text to display in waitbar
%   cmd_window_version[0] - use old command window version 
%   percent_increment [10] - percentage increment between successive updates
%   format_string ['%i %%'] - format string for output

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set default values
persistent h
if nargin>=3
    waitbar_text = varargin{1};
else
    waitbar_text = 'Please wait';
end;
if nargin>=4
    use_cmd_window_version = varargin{2};
else
    use_cmd_window_version = 0;
end;
if nargin>=5
    percent_increment = varargin{3};
else
    percent_increment = 10;
end;
if nargin>=6
    format_string = varargin{4};
else
    format_string = '%i %%';
end;


if ~use_cmd_window_version
    if ishandle(h)
        waitbar(current_count / total_count, h);
    else
        h = waitbar(current_count / total_count, waitbar_text);
    end;
    if (current_count == total_count) & ishandle(h)
        close(h);
    end;
else
    %work out if its time to display string
    nn = round((100 - current_count / total_count * 100) / percent_increment);
    yy = round((100 - nn * percent_increment) / 100 * total_count);
    if current_count  == yy
        %display string
        disp(sprintf(format_string,round(current_count / total_count * 100)));
    end;
end;

return;

