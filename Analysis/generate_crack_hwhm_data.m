%generate lookups for 2D S-matrix method
fn_clear;

N_nodes_per_wavelength = 40;
vL = 1;
vS = 0.5;

lambdaL = 1;
density = 1;
frequency = vL / lambdaL;

[lambda, mu] = fn_lame_from_velocities_and_density(vL, vS, 1);
[youngs_modulus, poissons_ratio] = fn_youngs_from_lame(lambda, mu);

crack_lengths = [0.05:0.05:5];
phi = linspace(-pi,pi,45);
L_in_cases = 1;
S_in_cases = 0;

hwhm = zeros(size(crack_lengths));
for ii = 1:length(crack_lengths)
    [S_LL, S_LS, S_SL, S_SS] = fn_s_matrices_for_crack_2d(youngs_modulus, poissons_ratio, density, frequency, N_nodes_per_wavelength, crack_lengths(ii), phi, phi, L_in_cases, S_in_cases);
    di = diag(S_LL);
    di = di(1:end-1);
    di2 = interpft(di, length(di) * 16);
    phi2 = linspace(-pi, pi, length(di2) + 1);
    phi2 = phi2(1: end - 1);
    di2 = abs(di2) / max(abs(di2));
    di2 = di2(1:round(length(di2)/4));
    phi2 = phi2(1:round(length(phi2)/4)) + pi;
    hwhm(ii) = phi2(max(find(di2 > 0.5)));
end

plot(crack_lengths, hwhm * 180 / pi, 'r.-');