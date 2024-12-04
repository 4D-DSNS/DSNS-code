function q = curve_to_q(p, toNormalize)
if nargin < 2
    toNormalize = 1;
end

[n,N] = size(p);
for i = 1:n
    v(i,:) = gradient(p(i,:),1/N);
end

for i = 1:N
    L(i) = sqrt(norm(v(:,i),'fro'));

    if L(i) > 0.00001
        q(:,i) = v(:,i)/L(i);
    else
        q(:,i) = 0*ones(n,1);
    end

end


if toNormalize
    q = q/sqrt(InnerProd_Q(q,q));
end
