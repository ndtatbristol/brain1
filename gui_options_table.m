function [h_table, h_fn_get_data, h_fn_set_data, h_fn_set_content, h_data_changed] = gui_options_table(h_parent, pos, pos_units, h_cb_data_changed)
h_table = uitable(h_parent, ...
    'ColumnEditable', [false], ...
    'RowName', [], ...
    'ColumnName', [], ...
    'Units', pos_units, ...
    'Position', pos, ...
    'CellSelectionCallback', @cb_cell_select, ...
    'Visible', 'off');

data = []; %data is the actual data held in the table with same fieldnames as content
content = []; %this describes what is going to go in the table (label strings, data types, constraints on values etc)
%the fieldnames in data are forced to match fieldnames in content info when
%content is specified via fn_set_content. If fn_set_data is called,
%only the fielnames in supplied data that match those in content will
%be used

fn_set_col_widths(h_table)

set(h_table, ...
    'Visible', 'on');

h_fn_get_data = @fn_get_data;

h_fn_set_data = @fn_set_data;

h_fn_set_content = @fn_set_content;

% h_fn_get_content = @fn_get_content;

h_data_changed = @fn_data_changed;

    function fn_data_changed(dummy)
        if ~isempty(data)
            h_cb_data_changed(data);
        end
    end

    function fn_set_content(new_content)
        content = new_content;
        data = []; %clear data
        info_fieldnames = fieldnames(content);
        for ii = 1:length(info_fieldnames)
            data = setfield(data, info_fieldnames{ii}, getfield(content, info_fieldnames{ii}, 'default'));
        end;
        fn_populate_table;
    end

%     function ct = fn_get_content(dummy)
%         ct = content;
%     end

    function x = fn_get_data(dummy)
        x = data;
    end

    function fn_set_data(x)
        if isempty(content)
            return;
        end
        data = x;
        %match data to fields in content
        data_fieldnames = fieldnames(data);
        %to do
        %******
        fn_populate_table;
    end

    function fn_populate_table
        %this fills the table according to content and uses current
        %data fields for values
        if isempty(content)
            return
        end
        info_fieldnames = fieldnames(content);
        val = cell(length(length(info_fieldnames)), 1);%this is where the strings to go into table are written
        for ii = 1:length(info_fieldnames)
            %add a row field to content as this is the way it is identified
            %from table click
            content = setfield(content, info_fieldnames{ii}, 'row', ii);
            s1 = getfield(content, info_fieldnames{ii}, 'label');
            x = getfield(data, info_fieldnames{ii});
            %now convert into string for display
            switch getfield(content, info_fieldnames{ii}, 'type')
                case 'bool'
                    if x
                        s2 = char(getfield(content, info_fieldnames{ii}, 'constraint', {1}));
                    else
                        s2 = char(getfield(content, info_fieldnames{ii}, 'constraint', {2}));
                    end;
                case 'constrained'
                    s2 = x;
                case 'int'
                    s2 = num2str(x);
                case 'double'
                    s2 = num2str(x / getfield(content, info_fieldnames{ii}, 'multiplier'));
                case 'string'
                    s2 = x;
                    
            end;
            val{ii, 1} = [s1, ': ', s2];
        end
        set(h_table, 'Data', val);
    end


    function cb_cell_select(h_table, eventdata)
    
        temp = get(h_table,'Data'); % Matlab suggested solution to removing selection
        set(h_table,'Data',[]);     % https://stackoverflow.com/questions/19634250/how-to-deselect-cells-in-uitable-how-to-disable-cell-selection-highlighting#19654513
        set(h_table,'Data', temp ); % Alternative in link rejected since it breaks GUI afterwards 
    
        ij = eventdata.Indices;
        if isempty(ij)
            return;
        end;
        ii = ij(1);
        %find the fieldname by looking through content to match row
        info_fieldnames = fieldnames(content);
        for jj = 1:length(info_fieldnames)
            if getfield(content, info_fieldnames{ii}, 'row') == ii;
                fname = info_fieldnames{ii};
                break
            end
        end
        
        info = getfield(content, fname);
        info.current = getfield(data, fname);
        
        switch info.type
            case 'int'
                val = fn_get_int_data(info);
            case 'double'
                info.current = info.current / info.multiplier;
                val = fn_get_double_data(info);
            case 'bool'
                if info.current
                    info.current = info.constraint{1};
                else
                    info.current = info.constraint{2};
                end
                val = fn_get_bool_data(info);
            case 'constrained'
                val = fn_get_constrained_data(info);
            case 'string'
                val = fn_get_string_data(info);
        end;
        if isempty(val)
            return;
        end
        data = setfield(data, fname, val);
        fn_populate_table;
        h_cb_data_changed(data);
    end

end

function val = fn_get_int_data(content)
val = [];
if iscell(content.constraint)
    min_val = eval(content.constraint{1});
    max_val = eval(content.constraint{2});
else
    min_val = content.constraint(1);
    max_val = content.constraint(2);
end
x = min_val - 1;
while x < min_val | x > max_val
    % Attempt to use newid.m which was written by Matlab to allow "Enter" keypress to activate OK/Cancel buttons
    try
        x = newid(sprintf([content.label, ' [%g to %g]'], [min_val, max_val]), ...
        'Input integer', 1, {num2str(content.current)});
    catch
        x = inputdlg(sprintf([content.label, ' [%g to %g]'], [min_val, max_val]), ...
        'Input integer', 1, {num2str(content.current)});
    end    
    if isempty(x)
        return;
    end;
    x = round(str2num(x{1}));
end;
val = x;
end

function val = fn_get_double_data(content)
val = [];
if iscell(content.constraint)
    min_val = eval(content.constraint{1});
    max_val = eval(content.constraint{2});
else
    min_val = content.constraint(1);
    max_val = content.constraint(2);
end
x = min_val - 1;

while x < min_val | x > max_val
    % Attempt to use newid.m which was written by Matlab to allow "Enter" keypress to activate OK/Cancel buttons
    try
        x = newid(sprintf([content.label, ' [%g to %g]'], [min_val, max_val] / content.multiplier), ...
            'Input float', 1, {num2str(content.current)});
    catch
        x = inputdlg(sprintf([content.label, ' [%g to %g]'], [min_val, max_val] / content.multiplier), ...
            'Input float', 1, {num2str(content.current)});
    end
    if isempty(x)
        return;
    end;
    x = str2num(x{1}) * content.multiplier;
end;
val = x;
end

function val = fn_get_bool_data(content)
val = [];
ii = find(strcmp(content.constraint, content.current));
[x,ok] = listdlg('ListString',content.constraint, 'SelectionMode', 'single', 'InitialValue', ii);
if ~ok
    return;
end
if x == 1
    val = 1;
else
    val = 0;
end
end

function val = fn_get_string_data(content)
val = [];
x = inputdlg(content.label, 'Input string', 1, {content.current});
if isempty(x)
    return;
end;
val = x{1};
end

function val = fn_get_constrained_data(content)
val = [];
ii = find(strcmp(content.constraint, content.current));
[x,ok] = listdlg('ListString',content.constraint, 'SelectionMode', 'single', 'InitialValue', ii);
if ~ok
    return;
end;
val = content.constraint{x};
end



function fn_set_col_widths(h_table)
p = getpixelposition(h_table);
set(h_table, ...
    'ColumnWidth', {p(3) * 10});
end

