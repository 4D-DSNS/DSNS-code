function [Theta,Phi,Psi] = spherharm(l,n)

theta = pi*[.01:n+1-.01]/(n+.02);
phi   = 2*pi*[0:n]/n;
Theta = repmat(theta,n+1,1);
Phi   = repmat(phi',1,n+1);

idx=1;
for j=1:l
    for k=1:j+1
        if (k-1==0)
            [Psi(:,:,idx),tmp]=spharm(j,k-1,Phi,Theta,[n+1 n+1],0);
            idx=idx+1;
        else
            [Psi(:,:,idx),Psi(:,:,idx+1)]=spharm(j,k-1,Phi,Theta,[n+1 n+1],0);
            idx=idx+2; 
        end
    end
end