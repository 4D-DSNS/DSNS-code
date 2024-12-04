function w = findphistarclosed(q2,Psi,b,Theta)

[a1,a2,a3,a4] = size(b);

Psi1=Psi(:,:,1:3);
Psi2=Psi(:,:,4:8);
Psi3=Psi(:,:,9:end);
c1=size(Psi1,3);
c2=size(Psi2,3);

idx=1;
for j=1:a4/2
    if (idx<=c1)
        divb(:,:,j)=-2*Psi1(:,:,idx);
        idx=idx+1;
    elseif (c1<idx && idx<=c1+c2)
        divb(:,:,j)=-6*Psi2(:,:,idx-c1);
        idx=idx+1;
    elseif (idx>c2)
        divb(:,:,j)=-12*Psi3(:,:,idx-c1-c2);
        idx=idx+1;
    end
end

[dq2du, dq2dv] = findgrad_closed(q2,Theta);

for k=1:a4/2
    for j=1:3
        expr11(:,:,j,k) = divb(:,:,k).*q2(:,:,j);
        expr21(:,:,j,k) = dq2du(:,:,j).*b(:,:,1,k)+dq2dv(:,:,j).*b(:,:,2,k);
    end
end
for k=(a4/2+1):a4
    for j=1:3
        expr11(:,:,j,k) = zeros(a1,a1);
        expr21(:,:,j,k) = dq2du(:,:,j).*b(:,:,1,k)+dq2dv(:,:,j).*b(:,:,2,k);
    end
end
for k=1:a4
    for j=1:3
        w(:,:,j,k) = expr11(:,:,j,k)+expr21(:,:,j,k);
        % w(:,:,j,k) = expr21(:,:,j,k);
    end
end

% fprintf("dq2du Values \t: %s \n",num2str(b(1,1,1, (a4/2+1):a4)));
% fprintf("dq2du Values \t: %s \n",num2str(b(1,1,2, 1:(a4/2))));
% fprintf("dq2dv Values \t");
% fprintf(num2str(b(:,:,2)));

% print(b(1,1,1,k))