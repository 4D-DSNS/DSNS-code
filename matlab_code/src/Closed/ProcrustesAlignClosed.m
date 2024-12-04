function [F2n,Ot,rho,T] = ProcrustesAlignClosed(q1,q2,F1,F2,sqrtmultfact1,sqrtmultfact2,Theta)

Ot = optimal_rot_surf1(F1,F2,Theta);
F2 = rotate3D(F2,Ot);
% Ot=eye(3);
rho = opt_scale_surf(q1,q2,sqrtmultfact1,sqrtmultfact2,Theta);
F2 = rho*F2;
% rho=1;
% T = opt_tran_surf(q1,q2,sqrtmultfact2,Theta);
T=zeros(3,1);
F2 = ApplyTran(F2,T);

% keyboard;

F2n=F2;
    