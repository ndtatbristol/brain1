function array = fn_create_annular_array(n, rc, re, gap)
% USAGE
%   array = array = fn_create_annular_array(n, rc, re, gap)
% SUMMARY
% Function to create the x, y & z co-ordinates of a one dimensional linear
% array in which the length and width of the array are defined even if it
% is later assumed that the length is infinite. The centre of the array will
% be the origin.
% 
% INPUTS - uses SI units throughout  (overall array centre will be at x=y=z=0, elements are number by row counting out from centre and then around each row starting with first element above x axis)
% n                 - m-element vector of number of elements around each ring
% rc                - m-element vector of radial positions of element centres
% re                - (m+1)-element vector of radial positions of element edges
% gap               - gap between elements in m

% OUTPUTS -  a structured variable whos main is defined by 'array'
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
% .element_type    - element type ['annular']

array.el_type = 'annular';

no_elements = sum(n);
array.el_xc = zeros(1, no_elements);
array.el_yc = zeros(1, no_elements);
array.el_zc = zeros(1, no_elements);
array.el_x1 = zeros(1, no_elements);
array.el_y1 = zeros(1, no_elements);
array.el_z1 = zeros(1, no_elements);
array.el_x2 = zeros(1, no_elements);
array.el_y2 = zeros(1, no_elements);
array.el_z2 = zeros(1, no_elements);

%do the centres
j = 1;
for i = 1:length(n)
    %create the angles
    dtheta = 2 * pi / n(i);
    theta = ([1:n(i)] - 0.5) * dtheta;
    array.el_xc(j: j + n(i) - 1) = rc(i) * cos(theta);
    array.el_yc(j: j + n(i) - 1) = rc(i) * sin(theta);
    if n(i) > 1
        %general case 
        %"1" point is at outer radius of element
        array.el_x1(j: j + n(i) - 1) = (re(i + 1) - gap / 2) * cos(theta);
        array.el_y1(j: j + n(i) - 1) = (re(i + 1) - gap / 2) * sin(theta);
        %"2" point is on the radial centreline at the +ve theta extremity
        array.el_x2(j: j + n(i) - 1) = rc(i) * cos(theta + dtheta / 2 - gap / rc(i) / 2);
        array.el_y2(j: j + n(i) - 1) = rc(i) * sin(theta + dtheta / 2 - gap / rc(i) / 2);
    else
        %special case for single round element (at centre of array)
        %"1" point is at 3oclock
        array.el_x1(j: j + n(i) - 1) = re(i + 1) - gap / 2;
        array.el_y1(j: j + n(i) - 1) = 0;
        
        array.el_x2(j: j + n(i) - 1) = 0;
        array.el_y2(j: j + n(i) - 1) = re(i + 1) - gap / 2;
    end
    j = j + n(i);
end    
    



%specify that all elements are both transmitter and reciever
array.el_tx = ones(1, no_elements);
array.el_rx = ones(1, no_elements);

%define co-ordinate vectors
% array.el_xc = [1:no_elements] * el_pitch;
% array.el_xc = array.el_xc - mean(array.el_xc);
% array.el_yc = zeros(size(array.el_xc));
% array.el_zc = zeros(size(array.el_xc));
% array.el_x1 = array.el_xc + el_width / 2;
% array.el_y1 = zeros(size(array.el_xc));
% array.el_z1 = zeros(size(array.el_xc));
% array.el_x2 = array.el_xc;
% array.el_y2 = array.el_yc + el_length / 2;
% array.el_z2 = zeros(size(array.el_xc));
return;
