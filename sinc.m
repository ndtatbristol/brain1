function y = sinc(x)
%our own sinc function as Matlab one is in a toolbox and not always
%available
ii = find(abs(x) < eps);
x(ii) = 1;
y = sin(pi * x) ./ (pi * x);
y(ii) = 1;
end