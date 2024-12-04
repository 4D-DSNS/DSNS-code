function ShowCorresp(F1,F2)

% load('E:\data\Geodesics\hand\hand2.mat');
% M1 = M;
% 
% load('E:\data\Geodesics\hand\hand3.mat');
% M2 = M;

for j=1:3
    M1(j,:,:)=F1(:,:,j)';
    M2(j,:,:)=F2(:,:,j)';
end

T = squeeze(M1(1, :, :));
[h w] = size(T);
shift = [2.5*max(T(:)), 0, 0];  


%=======================================================================
% Example 1: 
%=======================================================================
% Just displaying the models without a color map that shows the
% correspondence
% cmap = hsv(numel(T(:)));
% figure, hold on;
% surface(squeeze(M1(1, :, :)), squeeze(M1(2, :, :)), squeeze(M1(3, :, :)), ones( h, w)  );
% surface(squeeze(M2(1, :, :)) + shift(1), squeeze(M2(2, :, :)) + shift(2), squeeze(M2(3, :, :)) + shift(3),   ones( h, w)  );
% caxis([0 1]);

% cameratoolbar;
% axis off;
% axis equal;

% camlight left
% lighting phong
% shading 'flat';

% colormap([cmap; 1 0.95 0.9]);
% set(gcf, 'Renderer', 'zbuffer');
% set(gcf, 'Color', [1, 1, 1]);

% hold off;

%=======================================================================
% Example 2: 
%=======================================================================
% Displaying the models with a color map that shows the
% correspondences
figure(5);
cmap = hsv(w*h);
figure, hold on;
surface(squeeze(M1(1, :, :)), squeeze(M1(2, :, :)), squeeze(M1(3, :, :)));
surface(squeeze(M2(1, :, :)) + shift(1), squeeze(M2(2, :, :)) + shift(2), squeeze(M2(3, :, :)) + shift(3),squeeze(M1(3,:,:)));

cameratoolbar;
axis off;
axis equal;

camlight left
lighting phong
shading 'flat';

colormap([cmap; 1 0.95 0.9]);
set(gcf, 'Renderer', 'zbuffer');
set(gcf, 'Color', [1, 1, 1]);

hold off;