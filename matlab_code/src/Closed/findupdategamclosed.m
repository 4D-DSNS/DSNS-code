function gamupdate = findupdategamclosed(v,w,b,Theta)

[a1,a2,a3,a4] = size(w);
[a5,a6,a7,a8] = size(b);
n=a1-1;
dphi=2*pi/(n);
dtheta=(n*pi+pi-.02*pi)/(n^2+.02*n);

for k=1:a4
    for j=1:a3
        innp1(:,:,j,k)=v(:,:,j).*w(:,:,j,k);
    end
    innp2(:,:,k)=sum(innp1(:,:,:,k),3);
    innp(k)=sum(sum(innp2(:,:,k).*sin(Theta),2),1).*dtheta.*dphi;
end


gamupdate = zeros(a1,a2,2);
for k=1:a8
        gamupdate(:,:,1) = gamupdate(:,:,1) + innp(k)*b(:,:,1,k);
        gamupdate(:,:,2) = gamupdate(:,:,2) + innp(k)*b(:,:,2,k);
end