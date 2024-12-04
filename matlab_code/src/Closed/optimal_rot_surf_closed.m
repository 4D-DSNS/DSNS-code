function Ot = optimal_rot_surf_closed(q1, q2, Theta)

[n,t,d] = size(q1);
b = zeros(3,3);

for i=1:n
    for j=1:n
        a=squeeze(q1(i,j,:))*squeeze(q2(i,j,:))'*sin(Theta(i,j));        
        b=b+a;
    end
end
dphi  = (2*pi)/(n-1);
dtheta= (n*pi+pi-.02*pi)/(n^2+.02*n);
A     =  b*dphi*dtheta;

[U,S,V] = svd(A);
K       = [1,0,0;0,1,0;0,0,-1];

if det(A)> 0
    Ot = U*V';
else
    Ot = U*K*V';
end