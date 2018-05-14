function finished = plot_surf_2d(PS_db, p1, p2)
figure
% set(gcf, 'Position', [200, 200, 1300, 500])
%set(gcf, 'units','normalized', 'Position', [0.2, 0.2, 0.6, 0.4])
PS_db = PS_db';
c_black = [0 0 0]; c_red = [1 0 0]; c_blue = [0 0 1];  c_yellow = [1 1 0]; %c_green = [0 1 0];
f11 = c_black;
f12 = c_red;
f21 = c_blue;
f22 = c_yellow;
x1 = p1(1); y1 = p2(1); x2 = p1(end); y2 = p2(end);
% [x,y] = meshgrid(p1',p2');
[x,y] = meshgrid(p1',p2');
x = x(:); y = y(:);
fvc = zeros(length(x),3);
fvc(:,1) = 1/(x2-x1)/(y2-y1) * (f11(1)*(x2-x).*(y2-y) + f21(1)*(x-x1).*(y2-y) + f12(1)*(x2-x).*(y-y1) + f22(1)*(x-x1).*(y-y1));
fvc(:,2) = 1/(x2-x1)/(y2-y1) * (f11(2)*(x2-x).*(y2-y) + f21(2)*(x-x1).*(y2-y) + f12(2)*(x2-x).*(y-y1) + f22(2)*(x-x1).*(y-y1));
fvc(:,3) = 1/(x2-x1)/(y2-y1) * (f11(3)*(x2-x).*(y2-y) + f21(3)*(x-x1).*(y2-y) + f12(3)*(x2-x).*(y-y1) + f22(3)*(x-x1).*(y-y1));
pVert = [x, y]; 
pFaces = delaunayn(pVert);
subplot(121)

patch('Faces',pFaces,'Vertices',pVert,'FaceVertexCData',fvc,'FaceColor','interp','LineWidth',0.1,'LineStyle','none', 'FaceAlpha', 0.9);
%ylabel('size (\lambda)', 'FontSize', 18)
%xlabel('angle (\circ)', 'FontSize', 18)
%set(gca, 'FontSize',16)
ylabel('size (\lambda)')
xlabel('angle (\circ)')
%set(gca, 'FontSize',16)
axis tight
subplot(122)

for ii = 1:size(pFaces, 1)
    node_ref = pFaces(ii, 1);
    patch([PS_db(pFaces(ii, 1), 1), PS_db(pFaces(ii, 2), 1), PS_db(pFaces(ii, 3), 1)], ...
        [PS_db(pFaces(ii, 1), 2), PS_db(pFaces(ii, 2), 2), PS_db(pFaces(ii, 3), 2)],...
        [PS_db(pFaces(ii, 1), 3), PS_db(pFaces(ii, 2), 3), PS_db(pFaces(ii, 3), 3)], fvc(node_ref, :), 'LineStyle', 'none', 'FaceAlpha', 0.9);
    hold on
end
axis equal
%xlabel('PC 1', 'FontSize', 18)
%ylabel('PC 2', 'FontSize', 18)
%zlabel('PC 3', 'FontSize', 18)
%set(gca, 'FontSize',16)
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
%set(gca, 'FontSize',16)

view(3)

finished = 1;
end