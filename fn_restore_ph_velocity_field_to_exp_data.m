function exp_data = fn_restore_ph_velocity_field_to_exp_data(exp_data)
%SUMMARY
%	Restores legacy ph_velocity field to exp_data for backwards
%	compatibility with older imaging functions.
%INPUTS
%	exp_data - experimental data structure containing either 'material',
%	'vel_elipse', or 'ph_velocity' field.
%OUTPUTS
%	exp_data - experimental data structure with 'ph_velocity' field
%--------------------------------------------------------------------------

material = fn_material_from_exp_data(exp_data);
[exp_data.ph_velocity, ~, ~, ~]= fn_get_nominal_velocity(material.vel_spherical_harmonic_coeffs);
end