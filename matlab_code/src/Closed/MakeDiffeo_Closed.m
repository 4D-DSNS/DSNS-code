function [gamid] = MakeDiffeo_Closed(a,n,PlotDiffeo,b)

n = n-1;
theta = pi*[.01:n+1-.01]/(n+.02);
phi   = 2*pi*[0:n]/n;
Theta = repmat(theta,n+1,1);
Phi   = repmat(phi',1,n+1);

gamid(:,:,1) = Theta;
gamid(:,:,2) = Phi;

% for j=1:size(b,4)
%     gam1(:,:,1,j) = (a/j)*b(:,:,1,j);
%     gam1(:,:,2,j) = (a/j)*b(:,:,2,j);
% end
% 
% gam(:,:,1) = gamid(:,:,1) + sum(gam1(:,:,1,:),4);
% gam(:,:,2) = gamid(:,:,2) + sum(gam1(:,:,2,:),4);
% 
% if (PlotDiffeo)
%     figure(99); clf; hold on;
%     for i=1:n
%         plot(gam(i,:,1),gam(i,:,2));
%         plot(gam(:,i,1),gam(:,i,2));
%     end
% end