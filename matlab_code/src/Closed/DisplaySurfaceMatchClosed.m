function DisplaySurfaceMatchClosed(f1,f2,j)

global ShowFigures

% if ShowFigures==0
%    return
% end

[n,t,d] = size(f1);
c = max(max(f1(:,:,1)))+.5;

figure(j);
clf;
hold on;
a=plot3(f2(1,1,1),f2(1,1,2),f2(1,1,3),'y*','LineWidth',4);
b=plot3(f2(end,end,1),f2(end,end,2),f2(end,end,3),'g*','LineWidth',4);
c=plot3(f2(end,round(end/2),1),f2(end,round(end/2),2),f2(end,round(end/2),3),'k*','LineWidth',4);
d=plot3(f2(round(end/2),round(end/2),1),f2(round(end/2),round(end/2),2),f2(round(end/2),round(end/2),3),'c*','LineWidth',4);
e=plot3(f2(round(end/4),round(end/2),1),f2(round(end/4),round(end/2),2),f2(round(end/4),round(end/2),3),'m*','LineWidth',4);
f=plot3(f2(round(3*end/4),round(end/2),1),f2(round(3*end/4),round(end/2),2),f2(round(3*end/4),round(end/2),3),'b*','LineWidth',4);
h1 = surface(f1(:,:,1),f1(:,:,2),f1(:,:,3));
h2 = surface(f2(:,:,1),f2(:,:,2),f2(:,:,3),f1(:,:,3));
set(h1,'LineWidth',1);
set(h1,'FaceColor','interp','EdgeColor','r');
set(h2,'LineWidth',1);
set(h2,'FaceColor','interp');
axis equal;
axis off;
view([-30 30]);
lighting phong;
cameratoolbar;
% shading flat;
% colormap([.9 .9 .9]);