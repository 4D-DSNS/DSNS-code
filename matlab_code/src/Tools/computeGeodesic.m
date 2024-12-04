% 
% Compute the gedoesic between M1 and M2 - M1 and
% Returns the geodesic curve in the orginal space
% - Xgeod : nSamples * size(q1, 1) x size(q1, 2)
%
% It assumes that q1 and q2 are already registered
%
function Xgeod = computeGeodesic(M1, M2, doNormalizeScale, nSamples)

if nargin < 4
    nSamples = 7;
end

if nargin < 3
    doNormalizeScale = 1;
end

q1 = curve_to_q(M1, doNormalizeScale);
q2 = curve_to_q(M2, doNormalizeScale);

Xgeod = zeros(nSamples, size(q1, 1), size(q1, 2));

for i=1:nSamples
   
    t = (i-1.0)/(nSamples-1);    
    q = (1-t)*q1 + t*q2;
    
    M = (1-t)*M1(:, 1) + t*M2(:, 1);        % geodesic between the first surface on first curve and first surface on the second curve
    
    Xgeod(i, :, :) = q_to_curve(q, M);
    
end
    
    