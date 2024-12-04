function T = opt_tran_surf(q1,q2,sqrtmultfact2,Theta)

[a1,a2,a3]=size(q1);

n=a1;

dphi=(2*pi)/(n-1);
dtheta=(n*pi+pi-.02*pi)/(n^2+.02*n);

for j=1:3
    q1n(:,:,j)=sin(Theta).*q1(:,:,j);
    q2n(:,:,j)=sin(Theta).*q2(:,:,j);
end

q1bar=dphi*dtheta*squeeze(sum(sum(q1n,1),2));
q2bar=dphi*dtheta*squeeze(sum(sum(q2n,1),2));
sqrtmultfact2bar=dphi*dtheta*sum(sum(sqrtmultfact2.*sin(Theta),1),2);

% q1bar=(1/n^2)*squeeze(sum(sum(q1,1),2));
% q2bar=(1/n^2)*squeeze(sum(sum(q2,1),2));
% sqrtmultfact2bar=(1/n^2)*sum(sum(sqrtmultfact2,1),2);

T = (q1bar-q2bar)/sqrtmultfact2bar;

% for j=1:3
% %     if(T(j,1)<0.0001)
%         T(j,1)=round(T(j,1)*10)/10;
% %     end
% end

% T=squeeze(T);