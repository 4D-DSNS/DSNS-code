function DisplayGeod(F,n,phi,th)

% figure(5),h = surface(F(:,:,end,1),F(:,:,end,2),F(:,:,end,3));
% axis equal;
% v=axis;
% close;

% figure(n); clf; hold on,
cmap = hsv;
cmap = [cmap; 1 0.95 0.9; 0.7 0.7 0.7];
ncolors = size(cmap, 1);
[h, w, ~, ~] = size(F);

% dx = 1.25;
% for j=1:size(F, 3)
% %     subplot(1,7,j),
%     hh = surface(F(:,:,j,1)+dx*(j-1),F(:,:,j,2),F(:,:,j,3), ncolors * ones( h, w), 'SpecularExponent',100 + n);
%     axis equal;
%     axis off;
%     view([1 90]);
% %     camlight headlight;
% %     light;
% %     lighting phong;
% %     colormap([.5 .5 .5]);
% end
% cameratoolbar;

lighting phong
camlight('headlight')
shading 'flat';

colormap(cmap);

set(gcf, 'Renderer', 'zbuffer');
set(gcf, 'Color', [1, 1, 1]);

% for j=1:7
%     s=(j-1)/6;
%     F(:,:,j,:)=s*F2+(1-s)*F1;
% end

T = squeeze(F(:,:,1,1));
cmap = hsv(numel(T(:)));
figure(n); clf; hold on,
% 
for j=1:7
%     subplot(1,7,j),
    h = surface(F(:,:,j,1)+1.2*(j-1),F(:,:,j,2),F(:,:,j,3),F(:,:,1,3));
    % plot3(squeeze(F(:,ind,j,1))+1.2*(j-1),squeeze(F(:,ind,j,2)),squeeze(F(:,ind,j,3)),'k','LineWidth',3)
%     set(h,'FaceColor','interp');
%     shading flat;
%     axis(v);
    axis equal;
    axis off;
    view(phi,th);
%     cameratoolbar;
axis off;
axis equal;

% camlight left
% lightangle(phi,th)
lighting phong
shading 'flat';

colormap([cmap; 1 0.95 0.9]);
% colormap([.9 .9 .9])
% set(gcf, 'Renderer', 'zbuffer');
% set(gcf, 'Color', [1, 1, 1]);

%     colormap([.4 .4 .4]);
%     camlight left;
%     shading flat;
end
light
% figure(n+50); clf; hold on,

% for j=1:7
%     h = surface(F(:,:,j,3)+.7*(j-1),F(:,:,j,1),F(:,:,j,2));
% %     set(h,'FaceColor','interp');
% %     axis(v);
%     shading flat;
%     axis equal;
%     axis off;
%     view([0 0]);
%     colormap([.4 .4 .4]);
%     lighting phong;
%     camlight headlight;
%     view([25 0]);
% end

% for j=8:14
%     subplot(2,7,j),
%     h = surface(Fn(:,:,j-7,1),Fn(:,:,j-7,2),Fn(:,:,j-7,3));
%     set(h,'FaceColor','interp');
%     axis(v);
%     axis equal;
%     axis off;
%     view([0 90]);
% end