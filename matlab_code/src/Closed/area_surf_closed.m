function [A,A_tmp1,A_tmp2] = area_surf_closed(F)
% Test
[n,t,d] = size(F);

n=n-1;

theta = pi*[.01:n+1-.01]/(n+.02);
phi = 2*pi*[0:n]/n;
Theta = repmat(theta,n+1,1);
Phi = repmat(phi',1,n+1);
dphi=2*pi/(n);
dtheta=(n*pi+pi-.02*pi)/(n^2+.02*n);

for i=1:d
    [dfdu(:,:,i), dfdv(:,:,i)] = gradient(F(:,:,i),dtheta,dphi);
    for j=1:n+1
        for k=1:n+1
            dfdv(j,k,i) = dfdv(j,k,i)./sin(Theta(j,k));
        end
    end
end
for i=1:size(dfdu,1)
    for j=1:size(dfdv,2)
        A_tmp(i,j,:) = cross(dfdu(i,j,:),dfdv(i,j,:));
        A_tmp1(i,j) = norm(squeeze(A_tmp(i,j,:)));
        A_tmp2(i,j) = sqrt(A_tmp1(i,j));
    end
end

A = sum(sum(A_tmp1.*sin(Theta)))*dphi*dtheta;