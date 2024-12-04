% Projects and entire 4D surface onto a low dim space defined by the mean
% Mu and the eigenvectors eigenVects
%
% Parameters
% S  - 
% Mu - 
% eigenVects - 
%
% Output:
% M - 
%
% Copyright
% Hamid Laga
% 2019/10/16
%

%
function M = projectExpressionToLowDim(S, Mu, eigenVects)

resolution = [256, 256];

[n, resolution(1), resolution(2), ~] = size(S);

nModes = size(eigenVects, 2);

M = zeros(nModes, n);

for i=1:n,
    
    [M(:, i), ~] = project2PCABasis(squeeze(S(i, :,:,:)), Mu, eigenVects, resolution);
    
    %% For visualization
%     figure(2), clf;
%     surface(squeeze(Sn(:, :, 1)), squeeze(Sn(:, :, 2)), squeeze(Sn(:, :, 3))+.1, 'SpecularExponent', 100);
%     axis off equal;
%     cameratoolbar;
%     pause
end