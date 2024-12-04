function Fnew = center_surface_closed(F,multfact,A,Theta)

[n,t,d]=size(F);
dphi=2*pi/(n-1);
dtheta=(n*pi+pi-.02*pi)/(n^2+.02*n);

for j=1:d
    Fnew(:,:,j)=F(:,:,j)-(sum(sum(F(:,:,j).*multfact.*sin(Theta)))*dphi*dtheta)/A;
end