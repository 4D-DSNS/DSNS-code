function q = surface_to_q(f, multfact)

[n,t,d] = size(f);

% for i=1:h
%     [dfdu(:,:,i), dfdv(:,:,i)] = gradient(f(:,:,i),1/(n));
% end

% % for i=1:n
% %     dfdu(i,:,1) = gradient(f(i,:,1),1/(t));
% %     dfdu(i,:,2) = gradient(f(i,:,2),1/(t));
% %     dfdu(i,:,3) = gradient(f(i,:,3),1/(t));
% % end
% % for j=1:t
% %     dfdv(:,j,1) = gradient(f(:,j,1),1/(n));
% %     dfdv(:,j,2) = gradient(f(:,j,2),1/(n));
% %     dfdv(:,j,3) = gradient(f(:,j,3),1/(n));
% % end
% 
for i=1:n
    for j=1:t
%         A(i,j,:) = cross(dfdu(i,j,:),dfdv(i,j,:));
%         multfact(i,j) = sqrt(norm(squeeze(A(i,j,:)))); %sqrt(sqrt(A(i,j,1)^2+A(i,j,2)^2+A(i,j,3)^2));
        q(i,j,:) = multfact(i,j)*f(i,j,:);
%         q(i,j,2) = multfact(i,j)*f(i,j,2);
%         q(i,j,3) = multfact(i,j)*f(i,j,3);
    end
end