function innprod = vfinnerprod(b1,b2,Theta)

n=size(b1,1);
n=n-1;

t1=sum(sum(squeeze(b1(:,:,1)).*squeeze(b2(:,:,1)).*sin(Theta)))*(((n*pi+pi-.02*pi)/(n^2+.02*n))*(2*pi/(n)));
t2=sum(sum(squeeze(b1(:,:,2)).*squeeze(b2(:,:,2)).*sin(Theta)))*(((n*pi+pi-.02*pi)/(n^2+.02*n))*(2*pi/(n)));
innprod=t1+t2;