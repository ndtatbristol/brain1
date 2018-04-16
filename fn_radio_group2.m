function fn_radio_group(h_root, tag, must_select_one)
%get handles of all buttons in group
bset = tag(1:findstr(tag, '.'));
h = findall(h_root, '-regexp', 'Tag', [bset, '*']);
h_pressed = findall(h_root, 'Tag', tag);

% disable callbacks
for ii = 1:length(h)
    get_res(ii).fields = get(h(ii));
    if isfield(get_res(ii).fields, 'ClickedCallback')
        cb{ii} = get(h(ii), 'ClickedCallback');
        set(h(ii), 'ClickedCallback', []);
    end
    if isfield(get_res(ii).fields, 'Callback')
        cb{ii} = get(h(ii), 'Callback');
        set(h(ii), 'Callback', []);
    end
end;

%uncheck the buttons that were not clicked
for ii = 1:length(h)
    if h(ii) ~= h_pressed
        if isfield(get_res(ii).fields, 'State')
            set(h(ii), 'State', 'Off');
        end
        if isfield(get_res(ii).fields, 'Value')
            set(h(ii), 'Value', 0);
        end
    end
end

if must_select_one
    if isfield(h_pressed, 'State')
        set(h_pressed, 'State', 'On');
    end
    if isfield(h_pressed, 'Value')
        set(h_pressed, 'Value', 1);
    end
end;

% enable callbacks
for ii = 1:length(h)
    if isfield(get_res(ii).fields, 'ClickedCallback')
        set(h(ii), 'ClickedCallback', cb{ii});
    end
    if isfield(get_res(ii).fields, 'Callback')
        set(h(ii), 'Callback', cb{ii});
    end
end;

end
