%
% Resamples a diffeo to a different resolution
%
% Implemented by Hamid Laga
function gamcum0 = resampleGamma(gamcum, Norig)
    
[X, Y, Z] = spherical_to_cart_m(squeeze(gamcum(:,:,1)), squeeze(gamcum(:,:,2)));
gamtmp(:,:,1) = X;
gamtmp(:,:,2) = Y;
gamtmp(:,:,3) = Z;

gamtmp = MakeClosedGrid(gamtmp, Norig); 

[Thetan,Phin,~] = cartesian_to_sph1(squeeze(gamtmp(:,:, 1)), squeeze(gamtmp(:,:, 2)), squeeze(gamtmp(:,:, 3)));
gamcum0(:,:,1)  = Thetan;
gamcum0(:,:,2)  = Phin;