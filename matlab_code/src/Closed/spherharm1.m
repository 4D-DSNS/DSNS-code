function [Theta,Phi,Psi] = spherharm1(l,n)

theta = pi*[.01:n+1-.01]/(n+.02);
phi   = 2*pi*[0:n]/n;
Theta = repmat(theta,n+1,1);
Phi   = repmat(phi',1,n+1);

idx=1;
for j=1:l+1
    for k=1:j
        if (k-1==0)
            [Psi(:,:,idx),tmp]=spharm(j-1,k-1,Theta,Phi,[n+1 n+1],0);
            idx=idx+1;
        else
            [Psi(:,:,idx),Psi(:,:,idx+1)]=spharm(j-1,k-1,Theta,Phi,[n+1 n+1],0);
            idx=idx+2; 
        end
    end
end