function DisplaySurfaceClosed(f,Theta,i)

global ShowFigures

if ShowFigures==0
   return
end

figure(i);
clf;
hold on;
h = surface(f(:,:,1),f(:,:,2),f(:,:,3)); %,f(:,:,3));
% a=plot3(f(1,1,1),f(1,1,2),f(1,1,3),'y*','LineWidth',4);
% b=plot3(f(end,end,1),f(end,end,2),f(end,end,3),'g*','LineWidth',4);
% c=plot3(f(end,round(end/2),1),f(end,round(end/2),2),f(end,round(end/2),3),'k*','LineWidth',4);
% d=plot3(f(round(end/2),round(end/2),1),f(round(end/2),round(end/2),2),f(round(end/2),round(end/2),3),'c*','LineWidth',4);
% e=plot3(f(round(end/4),round(end/2),1),f(round(end/4),round(end/2),2),f(round(end/4),round(end/2),3),'m*','LineWidth',4);
% f=plot3(f(round(3*end/4),round(end/2),1),f(round(3*end/4),round(end/2),2),f(round(3*end/4),round(end/2),3),'b*','LineWidth',4);
set(gca,'FontSize',14);
% [A,multfact,tmp] = area_surf_closed(f);
% f = center_surface_closed(f,multfact,A,Theta);
% f = scale_surf(f,A);
% colorbar;
% shading flat;
% grid on;
% set(h,'LineWidth',2);
set(h,'FaceColor','interp');
axis off;
axis equal; %rot3d;
lighting phong;
view([-30 30]);
cameratoolbar;
% colormap([.9 .9 .9]);