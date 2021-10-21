function C = fn_EHM_stiffness_matrix_for_layered_media(layers, through_thickness_direction)
%SUMMARY
%   Computes equivalent homogenous medium (EHM) stiffness matrix for
%   layered media
%INPUTS
%   layers - vector of structures containing fields:
%   layers(ii).thickness - thickness
%   layers(ii).C - stiffness matrix for layer (Voigt notation)
%   through_thickness_direction - either index (1-3) or letter (x, y or z)
%   describing layer normal direction
%OUTPUTS
%   Overall stiffness matrix for EHM (Voigt notation)

%--------------------------------------------------------------------------
%for later use
voigt_index1 = [1, 2, 3, 2, 1, 1];
voigt_index2 = [1, 2, 3, 3, 3, 2];

%convert through_thickness_direction to numeric value
if ischar(through_thickness_direction)
    switch lower(through_thickness_direction)
        case 'x'
            through_thickness_direction = 1;
        case 'y'
            through_thickness_direction = 2;
        case 'z'
            through_thickness_direction = 3;
        otherwise
            through_thickness_direction = 0;
    end
end
if ~any(through_thickness_direction == [1:3])
    error('Invalid through-thickness direction');
end

%work out constant stress and constant strain direction indices
const_stress = [
    (voigt_index1(1:3) == through_thickness_direction) & ...
    (voigt_index2(1:3) == through_thickness_direction), ...
    (voigt_index1(4:6) == through_thickness_direction) | ...
    (voigt_index2(4:6) == through_thickness_direction)];
var_stress = ~const_stress;
const_strain = var_stress;
var_strain = const_stress;

const_stress_ii = find(const_stress);
const_strain_ii = find(const_strain);
var_stress_ii = find(var_stress);
var_strain_ii = find(var_strain);

%loop over layers and average appropriate matrices
a = zeros(3);
b = zeros(3);
c = zeros(3);
d = zeros(3);
total_thickness = sum([layers(:).thickness]);
for ii = 1:length(layers)
    if layers(ii).thickness == 0
        continue;
    end
    frac = layers(ii).thickness / total_thickness;
    A = layers(ii).C(var_stress_ii, var_strain_ii);
    B = layers(ii).C(var_stress_ii, const_strain_ii);
    E = layers(ii).C(const_stress_ii, var_strain_ii);
    D = layers(ii).C(const_stress_ii, const_strain_ii);
    
    a = a + frac * (eye(3) / E);
    b = b + frac * (A / E);
    c = c + frac * (E \ D);
    d = d + frac * (B - A * (E \ D));
end

C = zeros(6);
C(var_stress_ii, var_strain_ii) = b / a; % A*
C(var_stress_ii, const_strain_ii) = b * (a \ c) + d; % B*
C(const_stress_ii, var_strain_ii) = eye(3) / a;% C*
C(const_stress_ii, const_strain_ii) = (eye(3) / a) * c;% D*

end