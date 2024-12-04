function qnew=rotate3D(q, Ot)

if size(q, 1) ~=3,
    % q is hxwx3,
    [n,t,d]=size(q);

    for i=1:n
        for j=1:t
            qnew(i,j,:)=Ot*squeeze(q(i,j,:));
        end
    end
else
   % q is 3 x (hw)
   qnew = Ot * q;    
end