function theta_out = fn_wrap(theta_in)
%does the opposite of matlab wrap function and forces angles to lie between
%-pi and +pi
max_lim = pi;
min_lim = -pi;
theta_out = theta_in - floor((theta_in - min_lim) / (max_lim - min_lim)) * (max_lim - min_lim);
return;