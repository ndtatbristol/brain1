function new_struct = fn_set_default_fields(old_struct, default_struct)
%USAGE
%	new_struct = fn_set_default_fields(old_struct, default_struct);
%SUMMARY
%	Use to add default fields and values to a structured variable, such as
%	options for a function.
%AUTHOR
%	Paul Wilcox (Dec 2003)
%INPUTS
%	old_struct - original structured variable
%	default_struct - structured variable containing default fields and
%	values
%OUTPUTS
%	new_struct - updated structured variable. All existing fields and values
%	in old_struct will be preserved, but any fields found in default_struct
%	and their values will be added

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
new_struct = old_struct;
default_fieldnames = fieldnames(default_struct);
for ii=1:length(default_fieldnames)
	if ~isfield(new_struct, default_fieldnames{ii})
		new_struct = setfield(new_struct, default_fieldnames{ii}, getfield(default_struct, default_fieldnames{ii}));
	end;
end;
return;
