function test_options_table
fn_clear;
f = figure;
h = uipanel('BorderType','none',...
    'BackgroundColor', get(gcf, 'Color'),...
    'Units', 'normalized',...
    'Position', [0, 0, 0.25, 1],...
    'Parent', f);

for ii = 1:100
    data(ii).label = 'Something';
    data(ii).value = 17.2;
    data(ii).type = 'double';
    data(ii).constraint = [0, 20];
    data(ii).fieldname = 'x';
end

h_table = fn_create_options_table(data, h, [0, 0, 1, 1], 'normalized', @cb_function)
end

function cb_function(ops)
ops
end