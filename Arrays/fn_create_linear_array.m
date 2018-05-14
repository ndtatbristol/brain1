function array = fn_create_linear_array(no_elements, el_width, el_length, el_pitch, varargin);
% USAGE
%   array = fn_create_linear_array(no_elements, el_width, el_length, pitch, [comments, element_type])
% SUMMARY
% Function to create the x, y & z co-ordinates of a one dimensional linear
% array in which the length and width of the array are defined even if it
% is later assumed that the length is infinite. The centre of the array will
% be the origin.
% 
% INPUTS - uses SI units throughout
% no_elements      - number of elements in the array
% el_width         - width of each element
% el_length        - length of each element
% el_pitch         - distance between the leading edge of each element
% comments         - any further comments you wish to save. (as string)
% element_type     - element type 'rectangular' (default) or 'elliptical' 
%
% OUTPUTS -  a structured variable whos main is defined by 'array'
% .no_elements     - number of elements in the array
% .element_numbers - row vector of element numbers
% .el_width        - width of each element
% .el_length       - length of each element
% .pitch           - distance between the leading edge of each element
% .length          - total length of the array
% .el_tx           - row vector of transmitter indices
% .el_rx           - row vector of receiver indices
% 
% .el_xc           - x co-ords of the centre of each element
% .el_yc           - y co-ords of the centre of each element
% .el_zc           - z co-ords of the centre of each element
%
% .el_x1           - x co-ords of the centre of the shortest side of each element
% .el_y1           - y co-ords of the centre of the shortest side of each element
% .el_z1           - z co-ords of the centre of the shortest side of each element
%
% .el_x2           - x co-ords of the centre of the longest side of each element
% .el_y2           - y co-ords of the centre of the longest side of each element
% .el_z2           - z co-ords of the centre of the longest side of each element
%
% .comments        - additional comments made ['']
% .element_type    - element type ['rectangular']


% array.no_elements = no_elements;
% array.el_width = el_width;
% array.el_length = el_length;
% array.el_pitch = el_pitch;

if nargin>4
   array.comments = varargin{1};
else
end;

if nargin>5
   array.el_type = varargin{2};
else
   array.el_type = 'rectangular';
end;

array.element_numbers = [1:no_elements];

%specify that all elements are both transmitter and reciever
array.el_tx = ones(1, no_elements);
array.el_rx = ones(1, no_elements);

%define co-ordinate vectors
array.el_xc = [1:no_elements] * el_pitch;
array.el_xc = array.el_xc - mean(array.el_xc);
array.el_yc = zeros(size(array.el_xc));
array.el_zc = zeros(size(array.el_xc));
array.el_x1 = array.el_xc + el_width / 2;
array.el_y1 = zeros(size(array.el_xc));
array.el_z1 = zeros(size(array.el_xc));
array.el_x2 = array.el_xc;
array.el_y2 = array.el_yc + el_length / 2;
array.el_z2 = zeros(size(array.el_xc));
return;