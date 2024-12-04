function Fnew = scale_surf(F,A)

[n,t,d]=size(F);

for i=1:n
    for j=1:n
        Fnew(i,j,:) = F(i,j,:)/sqrt(A);
    end
end