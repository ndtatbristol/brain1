function material = fn_material_from_exp_data(exp_data)
%SUMMARY
%	Returns new format 'material' structure from exp_data structure, even if
%	latter is in old format and contains 'ph_velocity' or 'vel_elipse' rather 
%	than 'material' field.

%--------------------------------------------------------------------------
if isfield(exp_data, 'material')
    material = exp_data.material;
elseif isfield(exp_data, 'vel_elipse')
	%legacy files with 'vel_elipse' field
    lebedev_quality = 3;
    material.vel_spherical_harmonic_coeffs = fn_spherical_harmonics_for_elliptical_profile(exp_data.vel_elipse(1), exp_data.vel_elipse(1), exp_data.vel_elipse(2), lebedev_quality);
elseif isfield(exp_data, 'ph_velocity')
	%legacy files with 'ph_velocity' field
    material.vel_spherical_harmonic_coeffs = exp_data.ph_velocity; %legacy files with ph_velocity field
else
    error('No valid velocity description found');
end

end