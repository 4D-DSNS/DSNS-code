function [Psi,b] = formbasiselements(n,Theta,Phi)

N(1)=calcN(1,1);
b(:,:,1,2)=-N(1).*cos(Phi).*cos(Theta);
b(:,:,2,2)=N(1).*sin(Phi);
Psi(:,:,2)=-N(1).*cos(Phi).*sin(Theta);
N(2)=calcN(1,0);
b(:,:,1,1)=-N(2).*sin(Theta);
b(:,:,2,1)=zeros(n+1,n+1);
Psi(:,:,1)=N(2).*cos(Theta);
N(3)=calcN(1,-1);
b(:,:,1,3)=(N(3)/2).*sin(-Phi).*cos(Theta);
b(:,:,2,3)=-(N(3)/2).*cos(-Phi);
Psi(:,:,3)=(N(3)/2).*sin(-Phi).*sin(Theta);
N(4)=calcN(2,2);
b(:,:,1,6)=6*N(4).*cos(2.*Phi).*sin(Theta).*cos(Theta);
b(:,:,2,6)=-6*N(4).*sin(2.*Phi).*sin(Theta);
Psi(:,:,6)=3*N(4).*cos(2.*Phi).*(sin(Theta)).^2;
N(5)=calcN(2,1);
b(:,:,1,5)=-3*N(5).*cos(Phi).*((cos(Theta).^2)-(sin(Theta).^2));
b(:,:,2,5)=3*N(5).*sin(Phi).*cos(Theta);
Psi(:,:,5)=-3*N(5).*cos(Phi).*cos(Theta).*sin(Theta);
N(6)=calcN(2,0);
b(:,:,1,4)=-3*N(6).*cos(Theta).*sin(Theta);
b(:,:,2,4)=zeros(n+1,n+1);
Psi(:,:,4)=(N(6)/2).*(3.*(cos(Theta)).^2-ones(n+1,n+1));
N(7)=calcN(2,-1);
b(:,:,1,7)=(N(7)/2).*sin(-Phi).*((cos(Theta).^2)-(sin(Theta).^2));
b(:,:,2,7)=-(N(7)/2).*cos(-Phi).*cos(Theta);
Psi(:,:,7)=(N(7)/2).*sin(-Phi).*cos(Theta).*sin(Theta);
N(8)=calcN(2,-2);
b(:,:,1,8)=(N(8)/4).*sin(-2.*Phi).*sin(Theta).*cos(Theta);
b(:,:,2,8)=-(N(8)/4).*cos(-2.*Phi).*sin(Theta);
Psi(:,:,8)=(N(8)/8).*sin(-2.*Phi).*(sin(Theta).^2);
N(9)=calcN(3,3);
b(:,:,1,9)=-45*N(9).*(sin(Theta).^2).*cos(Theta).*cos(3*Phi);
b(:,:,2,9)=45*N(9).*(sin(Theta).^3).*sin(3*Phi);
Psi(:,:,9)=-15*N(9).*(sin(Theta).^3).*cos(3*Phi);
N(10)=calcN(3,2);
b(:,:,1,10)=15*N(10).*(2*sin(Theta).*cos(Theta).^2-sin(Theta).^3).*cos(2*Phi);
b(:,:,2,10)=-30*N(10).*cos(Theta).*(sin(Theta).^2).*sin(2*Phi);
Psi(:,:,10)=15*N(10).*cos(Theta).*(sin(Theta).^2).*cos(2*Phi);
N(11)=calcN(3,1);
b(:,:,1,11)=-(3*N(11)/2).*(5*cos(Theta).^3-cos(Theta)-10*cos(Theta).*sin(Theta).^2).*cos(Phi);
b(:,:,2,11)=(3*N(11)/2).*(5*cos(Theta).^2-1).*sin(Theta).*sin(Phi);
Psi(:,:,11)=-(3*N(11)/2).*(5*cos(Theta).^2-1).*sin(Theta).*cos(Phi);
N(12)=calcN(3,0);
b(:,:,1,12)=(N(11)/2).*(3*sin(Theta)-15*cos(Theta).^2.*sin(Theta));
b(:,:,2,12)=zeros(n+1,n+1);
Psi(:,:,12)=(N(12)/2).*(5*cos(Theta).^3-cos(Theta));
N(13)=calcN(3,-1);
b(:,:,1,13)=(N(13)/8).*(5*cos(Theta).^3-cos(Theta)-10*cos(Theta).*sin(Theta).^2).*sin(-Phi);
b(:,:,2,13)=-(N(13)/8).*(5*cos(Theta).^2-1).*sin(Theta).*cos(-Phi);
Psi(:,:,13)=(N(13)/8).*(5*cos(Theta).^2-1).*sin(Theta).*sin(-Phi);
N(14)=calcN(3,-2);
b(:,:,1,14)=(N(14)/8).*(2*sin(Theta).*cos(Theta).^2-sin(Theta).^3).*sin(-2*Phi);
b(:,:,2,14)=-(N(14)/4).*cos(Theta).*sin(Theta).^2.*cos(-2*Phi);
Psi(:,:,14)=(N(14)/8).*cos(Theta).*sin(Theta).^2.*sin(-2*Phi);
N(15)=calcN(3,-3);
b(:,:,1,15)=(N(15)/16).*sin(Theta).^2.*cos(Theta).*sin(-3*Phi);
b(:,:,2,15)=-(N(15)/16).*sin(Theta).^3.*cos(-3*Phi);
Psi(:,:,15)=(N(15)/48).*sin(Theta).^3.*sin(-3*Phi);

for j=1:size(b,4)
    b(:,:,1,j+15) = b(:,:,2,j);
    b(:,:,2,j+15) = -b(:,:,1,j);
end