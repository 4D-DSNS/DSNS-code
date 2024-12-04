function Sn = reconstructSurface(Cn, Mu, eigenVectors, resolution)

nModes = length(Cn);

Sn = Mu;
for i=1:nModes
    Sn = Sn + Cn(i) * eigenVectors(:, i);
end
Sn = reshape(Sn, [resolution(1), resolution(2), 3]); 