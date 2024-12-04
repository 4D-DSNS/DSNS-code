% l   - oder of spherical harmonics (5 or 7 is usually good)
% n   - is the surface resolution   (50 if 50x50 grid)
function [Theta,Phi,Psi,b] = spherharmbasis1(l,n)

n = n-1;

theta = pi*[.01:n+1-.01]/(n+.02);
phi   = 2*pi*[0:n]/n;
Theta = repmat(theta,n+1,1);
Phi   = repmat(phi',1,n+1);

[Theta,Phi,Psi1] = spherharm1(l,n);
Psi              = Psi1(:,:,2:end);

[n,t,d] = size(Psi);
n       = n-1;

dphi   = (2*pi)/(n);
dtheta = (n*pi+pi-.01*pi)/(n^2+.02*n);
 
for j=1:d
    lenpsi(j)=sum(sum(squeeze(Psi(:,:,j)).*squeeze(Psi(:,:,j)).*sin(Theta)))*(dphi*dtheta);
end

eps = .1;

for j=1:d
    [dpsidtheta(:,:,j),~] = gradient(Psi(:,:,j),dtheta,dphi);
    [~,dpsidphi(:,:,j)]   = gradient(Psi(:,:,j),dtheta,dphi);
    b1(:,:,1,j) = dpsidtheta(:,:,j);
    for i=1:n+1
        for k=1:n+1
            b1(i,k,2,j) = dpsidphi(i,k,j);
        end
    end
end

b = b1;
k = size(b,4);
for j=(k+1):(2*k)
    b(:,:,1,j) = b1(:,:,2,j-k);
    b(:,:,2,j) = -b1(:,:,1,j-k);
end

[a1,a2,a3,a4] = size(b);

for j=1:a4
    blen(j) = vfinnerprod(b(:,:,:,j),b(:,:,:,j),Theta);
    b(:,:,:,j) = b(:,:,:,j)/sqrt(blen(j));
end

for j=1:d
    Psi(:,:,j) = Psi(:,:,j)/sqrt(blen(j));
end
