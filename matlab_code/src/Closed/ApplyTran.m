function qnew = ApplyTran(q,T)

[n,t,d]=size(q);

for i=1:n
    for j=1:t
        qnew(i,j,:)=squeeze(q(i,j,:))+T;
    end
end