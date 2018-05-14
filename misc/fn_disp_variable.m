function fn_disp_variable(x, options)
%SUMMARY
%   Displays summary of contents of variable x, which can be a structure.
%AUTHOR
%   Paul Wilcox (2008)
%USAGE
%   fn_disp_variable(x, options)
%INPUTS
%   x - variable of interest
%   options - display options
%OUTPUTS
%   none (output is to command window)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

default_options.indent_spaces = 2;
default_options.start_str = '';
options = fn_set_default_fields(options, default_options);
indent_str = repmat(' ', 1, options.indent_spaces);

if isstruct(x)
    fn_show_fields(x, options.start_str, indent_str);
else
    fn_display_variable(options.start_str, indent_str, 'value: ', x);
end;
return;

function fn_show_fields(x, start_str, indent_str);
fds = fieldnames(x);
for ii = 1:length(fds)
    z = getfield(x, fds{ii});
    fn_display_variable(start_str, indent_str, [fds{ii}, ': '], z)
end;
return;

function fn_display_variable(start_str, indent_str, name_str, z)
    if isempty(z)
        disp([start_str, indent_str, name_str, '[]']);
    else
        switch class(z)
            case 'double'
                fmt_str = '%g';
                if isscalar(z)
                    disp([start_str, indent_str, name_str, sprintf(fmt_str, z)]);
                else
                    if length(z(:)) > 2
                        mid_str = ' ... ';
                    else
                        mid_str = ', ';
                    end;
                    end_str = [' ['];
                    sz = size(z);
                    for ii = 1:length(sz)
                        end_str = [end_str, sprintf('%g', sz(ii))];
                        if ii < length(sz)
                            end_str = [end_str, 'x'];
                        end;
                    end;
                    end_str = [end_str, sprintf(' = %g elements]', length(z(:)))];
                    disp([start_str, indent_str, name_str, sprintf([fmt_str, mid_str, fmt_str], z(1), z(end)), end_str]);
                end;
            case 'char'
                disp([start_str, indent_str, name_str, '''', z, '''']);
            case 'struct'
                disp([start_str, indent_str, name_str, sprintf('[%g element structure array]', length(z))]);
                for ii = 1:length(z)
                    fn_show_fields(z(ii), [start_str, indent_str], indent_str);
                end;
            otherwise
                disp([start_str, indent_str, name_str, 'unknown type']);
        end;
    end;
return;