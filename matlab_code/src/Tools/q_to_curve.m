function p = q_to_curve(q, transVect)
%
% transVect - is the translation of the expression
%
%

if nargin<2
    transVect = zeros(size(q, 1), 1);
end

[n,T] = size(q);

for i = 1:T
    qnorm(i) = norm(q(:,i));
end


for i = 1:n
    p(i,:) = [ cumtrapz( q(i,:).*qnorm )/(T) ] ;
end

size(p)
size(transVect)
% T
% pause
p = p + repmat(transVect, 1, T);