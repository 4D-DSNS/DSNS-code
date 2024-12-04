% Composition of diffeomorphisms of the sphere to itself
% gamcum = gam1 o gam0
%
% Implemented by Hamid Laga
%
function gamcum = ComposeDiffeos(gam0, gam1, Theta, Phi, toDebug)

if nargin < 5,
    toDebug = 0;
end

[n,t,d] = size(gam0);

[X, Y, Z] = spherical_to_cart_m(Theta, Phi);
f(:,:,1) = X;
f(:,:,2) = Y;
f(:,:,3) = Z;

fnew = Apply_Gamma_Surf_Closed(f, Theta, Phi, gam0);

if toDebug,
    figure(3323); clf;
    surface(fnew(:,:,1),fnew(:,:,2),fnew(:,:,3));
    axis equal;
    cameratoolbar;
end

fnew = Apply_Gamma_Surf_Closed(fnew, Theta, Phi, gam1);
if toDebug,
    figure(3324); clf;
    surface(fnew(:,:,1),fnew(:,:,2),fnew(:,:,3));
    axis equal;
    cameratoolbar;
end


[Thetan, Phin] = cartesian_to_sph_m(squeeze(fnew(:,:,1)), squeeze(fnew(:,:,2)), squeeze(fnew(:,:,3)));
gamcum(:,:,1) = Thetan;
gamcum(:,:,2) = Phin;