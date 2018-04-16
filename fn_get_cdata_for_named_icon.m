function cdata = fn_get_cdata_for_named_icon(icons, name)
cdata = [];
for ii = 1:length(icons)
    if strcmpi(icons(ii).name, name)
        cdata = icons(ii).cdata;
        return;
    end;
end
end