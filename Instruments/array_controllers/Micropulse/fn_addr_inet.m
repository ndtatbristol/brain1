function ip_number = fn_addr_inet(ip_string)
temp = sscanf(ip_string, '%i.%i.%i.%i');
temp = char(strrep(cellstr(fliplr(char(dec2hex(fliplr(temp')),'xx'))),' ','0'));
temp = temp'; 
ip_number = hex2dec(temp(1:8));
return