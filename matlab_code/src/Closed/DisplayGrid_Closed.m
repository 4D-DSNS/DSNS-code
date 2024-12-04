function DisplayGrid_Closed(gamp,j)

[n,t,d] = size(gamp);
n=n-1;
n1=n+2;

theta = pi*[1:n1-1]/n1;
phi   = 2*pi*[0:n]/n;
Theta = repmat(theta,n+1,1);
Phi   = repmat(phi',1,n+1);

figure(j); clf; hold on;

f(:,:,1)=sin(Theta).*sin(Phi);
f(:,:,2)=sin(Theta).*cos(Phi);
f(:,:,3)=cos(Theta);
fnew = Apply_Gamma_Surf_Closed(f,Theta,Phi,gamp);



f(:,:,1)=sin(squeeze(gamp(:,:,1))).*sin(squeeze(gamp(:,:,2)));
f(:,:,2)=sin(squeeze(gamp(:,:,1))).*cos(squeeze(gamp(:,:,2)));
f(:,:,3)=cos(squeeze(gamp(:,:,1)));

clf;
hold on;
set(gca,'FontSize',14);
g = surface(f(:,:,1),f(:,:,2),f(:,:,3));
% set(g,'LineWidth',1)
% set(g,'FaceColor','interp','EdgeColor','b');
% h = surface(fnew(:,:,1),fnew(:,:,2),fnew(:,:,3));
grid on;
% set(h,'LineWidth',1.5);
% set(h,'FaceColor','interp','EdgeColor','r');
% axis equal off;
% colormap([.9 .9 .9]);
% view([-30 0]);