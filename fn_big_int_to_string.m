function s = fn_big_int_to_string(n, varargin)
%USAGE
%   s = fn_big_int_to_string(n, [str_len = 0, sep_char = ',', group_size = 3])
%SUMMARY
%   Converts integer number n into a text string with digits bunched in
%   groups of group_size separated by sep_char and optionally padded with
%   leading space to make up to length str_len.
if nargin < 2
    str_len = 0;
else
    str_len = varargin{1};
end
if nargin < 3
    sep_char = ',';
else
    sep_char = varargin{2};
end
if nargin < 4
    group_size = 3;
else
    group_size = varargin{3};
end
s = fliplr(sprintf([repmat('%c', 1, group_size), sep_char], fliplr(sprintf('%i',n))));
if s(1) == sep_char
    s(1) = [];
end
if s(1) == '-' && s(2) == sep_char
    s(2) = [];
end
if str_len == 0
    return
end
if length(s) < str_len
    s = [repmat(' ', 1, str_len - length(s)), s];
end