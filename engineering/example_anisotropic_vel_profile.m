%Example anisotopric velocity profiles

%Values for CMSX-4 from Kit Lane's thesis
c11 = 235e9;
c12 = 142e9;
c44 = 131e9;
rho = 8720;
C = [c11, c12, c12, 0, 0, 0
    c12, c11, c12, 0, 0, 0
    c12, c12, c11, 0, 0, 0
    0, 0, 0, c44, 0, 0
    0, 0, 0, 0, c44, 0
    0, 0, 0, 0, 0, c44];

a = linspace(0, 2*pi, 361)';
n = [cos(a), sin(a), zeros(361,1)];

[vph, vgr, p] = fn_anisotropic_vel_profile(C, rho, n);

figure;

c = 'rgb';
subplot(1,2,1);
for m = 1:3
    plot(vph(:,1,m), vph(:,2,m), c(m));
    hold on;
end
axis equal;
title('Phase velocity');

subplot(1,2,2);
for m = 1:3
    plot(vgr(:,1,m), vgr(:,2,m), c(m));
    hold on;
end
axis equal;
title('Group velocity');
