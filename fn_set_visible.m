function res = fn_set_visible(h, val)
if val
    set(h, 'Visible', 'On');
else
    set(h, 'Visible', 'Off');
end;
end