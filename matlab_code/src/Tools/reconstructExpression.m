% Reconstructs an entire 4D surface from its PCA rep
%
% Parameters
% M  -  PCA params
% Mu - 
% eigenVects - 
%
% Output:
% S - reconstructedsurface
%
% Copyright
% Hamid Laga
% 2019/10/16
function S = reconstructExpression(M, Mu, eigenVects, resolution)

n = size(M, 2);
S = zeros(n, resolution(1), resolution(2), 3); 

for i=1:n
    S(i, :,:,:) = reconstructSurface(M(:, i), Mu, eigenVects, resolution);
end