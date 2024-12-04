% Projecting a surface onto its PCA basis
% 
% Output
% Cn - the coefficients (i.e., the coordinates in the the new PCA basis)
% Sn - the reconstructed surface
% 
function [Cn, Sn] = project2PCABasis(S, Mu, eigenVectors, resolution, toVisualize)

if nargin < 5
    toVisualize = 0;
end

if nargin < 4
    resolution = [256, 256];
end

nModes = size(eigenVectors, 2);

%% Projection
Cn = eigenVectors(:, 1:nModes)' * (S(:) - Mu);

%% Reconstruct the surface from the projection
 %% Mu + eigenVectors(:, 1:nModes) * Cn(:)

Sn = reconstructSurface(Cn, Mu, eigenVectors, resolution);


% visualization
if toVisualize
    
    M1 = S;
    if (~ismatrix(M1)) % < 3)
        M1 = reshape(M1, [resolution(1), resolution(2), 3]);
    end
    
    M2 = Sn; 
    if (~ismatrix(M2))
        M2 = reshape(M2, [resolution(1), resolution(2), 3]);
    end
    
    if toVisualize
        figure(1), clf;
        surface(squeeze(M1(:, :, 1)), squeeze(M1(:, :, 2)), squeeze(M1(:, :, 3)), ...
                 'SpecularExponent',100);
        axis off equal;
        cameratoolbar;
        
        figure(2), clf;
        surface(squeeze(M2(:, :, 1)), squeeze(M2(:, :, 2)), squeeze(M2(:, :, 3)), ...
                 'SpecularExponent', 100);
        axis off equal;
        cameratoolbar;
        pause
    end
end