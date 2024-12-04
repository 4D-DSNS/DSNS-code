function [I_phi, I_theta] = sphCoord2Ind(landmarksPhi, landmarksTheta, N )

% [Theta, Phi] = getSphericalGrid(N);
theta = pi*[0:N]/N;
Theta = repmat(theta,N+1,1);

phi = 2*pi*[0:N]/(N);
Phi = repmat(phi',1, N+1);

th = Theta(1, :);
phi= Phi(:, 1)';

% Phi index  (Row index in Sebastian's convention)
I_phi = floor(landmarksPhi / phi(2))+1; 

% Theta index  (column in Sebastian's convention)
I_theta = floor( landmarksTheta / th(2))+1; 