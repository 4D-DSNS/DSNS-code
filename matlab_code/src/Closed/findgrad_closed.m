function [dfdu, dfdv] = findgrad_closed(f,Theta)

[n,t,d] = size(f);
% fprintf('%d %d %d\n',n,t,d);cl

for i=1:d
    [dfdu(:,:,i), dfdv(:,:,i)] = gradient(f(:,:,i),(n*pi-.02*pi)/((n-1)^2+.02*(n-1)),2*pi/(n-1));
    dfdk(:,:,i) = dfdv(:,:,i);
    for j=1:n
        for k=1:n
            dfdv(j,k,i)=dfdv(j,k,i)./sin(Theta(j,k));
        end
    end
end

% sz_dfdu = size(dfdu);
% sz_dfdv = size(dfdv);

% fprintf('%d %d %d %d %d %d\n',sz_dfdu(1),sz_dfdu(2),sz_dfdu(3), sz_dfdv(1), sz_dfdv(2), sz_dfdv(3));

% % Visualization of dfdu
% figure(97);
% for i = 1:d
%     subplot(1, d, i);
%     imagesc(dfdu(:,:,i));
%     colorbar;
%     title(['dfdu(:,:,', num2str(i), ')']);
% end

% % Visualization of dfdv
% figure(96);
% for i = 1:d
%     subplot(1, d, i);
%     imagesc(dfdv(:,:,i));
%     colorbar;
%     title(['dfdv(:,:,', num2str(i), ')']);
% end

% % Visualization of dfdk
% figure(95);
% for i = 1:d
%     subplot(1, d, i);
%     imagesc(dfdk(:,:,i));
%     colorbar;
%     title(['dfdk(:,:,', num2str(i), ')']);
% end

% [n, ~, d] = size(f);

% % Assume f is a vector field with 2 or 3 components (d=2 or d=3)
% % For 3 components, we visualize only two for a 2D quiver plot
% % Here, we choose the first two components for the u and v directions

% % If f has 3 components, we'll only use the first two for visualization
% if d == 3
%     U = f(:,:,1);
%     V = f(:,:,2);
% elseif d == 2
%     U = f(:,:,1);
%     V = f(:,:,2);
% else
%     error('f must have 2 or 3 components.');
% end

% % Create a meshgrid for u and v directions
% [u, v] = meshgrid(linspace(0, 2*pi, n), linspace(0, pi, n));

% % Visualization using quiver
% figure(98);
% quiver(u, v, U, V);
% title('Square Root Normal Fields with Arrows');
% xlabel('u direction');
% ylabel('v direction');
% axis tight;