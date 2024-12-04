function [gamnew] = updategam(gamupdate,gamid,eps)

% disp("Calling this function");
[a1,a2,a3] = size(gamid);
th = pi*[.01:a1-.01]/(a1-1+.02);

% const=th(end)-th(1);
% 
for k=1:a3
    gamnew(:,:,k) = gamid(:,:,k) + eps.*gamupdate(:,:,k);
end

for i=1:a1
    for j=1:a1
        if(gamnew(i,j,1)<th(1))
            gamnew(i,j,1)=th(1)+(th(1)-gamnew(i,j,1));
%             gamnew(i,j,2)=gamnew(i,j,2)-pi;
        elseif(gamnew(i,j,1)>th(end))
            gamnew(i,j,1)=th(end)-(gamnew(i,j,1)-th(end));
%             gamnew(i,j,2)=gamnew(i,j,2)-pi;
        end
    end
end

% gamnew(end,:,2)=gamnew(1,:,2);

% keyboard;