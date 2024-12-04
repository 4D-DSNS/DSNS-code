function rho = opt_scale_surf(q1,q2,sqrtmultfact1,sqrtmultfact2,Theta)

[a1,a2,a3]=size(q1);

n=a1;

% for j=1:3
%     q1n(:,:,j)=sin(Theta).*q1(:,:,j);
%     q2n(:,:,j)=sin(Theta).*q2(:,:,j);
% end
% 
% dphi=(2*pi)/(n-1);
% dtheta=(n*pi+pi-.02*pi)/(n^2+.02*n);
% 
% q1bar = dphi*dtheta*sum(sum(q1n,1),2);
% q2bar = dphi*dtheta*sum(sum(q2n,1),2);

q1bar = (1/(a1*a2))*sum(sum(q1,1),2);
q2bar = (1/(a1*a2))*sum(sum(q2,1),2);

for i=1:a1
    for j=1:a2
        tmp1(i,j,:)=squeeze(q1(i,j,:)-sqrtmultfact1(i,j)*q1bar);
        tmp2(i,j,:)=squeeze(q2(i,j,:)-sqrtmultfact2(i,j)*q2bar);
    end
end

innprod1=sum(sum(sum(tmp1.*tmp2,3).*sin(Theta))).*((n*pi+pi-.02*pi)/(n^2+.02*n)).*(2*pi/(n-1));
innprod2=sum(sum(sum(tmp2.*tmp2,3).*sin(Theta))).*((n*pi+pi-.02*pi)/(n^2+.02*n)).*(2*pi/(n-1));

% innprod1=sum(sum(sum(tmp1.*tmp2,3))).*(1/n^2);
% innprod2=sum(sum(sum(tmp2.*tmp2,3))).*(1/n^2);


rho=sqrt(innprod1/innprod2);

% .*sin(Theta)
% ((n*pi+pi-.02*pi)/(n^2+.02*n)).*(2*pi/(n-1)