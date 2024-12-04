%
% Implemented by Hamid
% Takes as input the qs not the surfaces
%
function [q2new, gamp] = ReParamclosed_SRNFs(q1,q2, ...
                                    b, Theta, Phi, Psi,  ...
                                    itermax,eps)

[n,t,d] = size(q1);
gamid   = MakeDiffeo_Closed(0,n,0,b);
gamp    = gamid;


H(1) = Calculate_Distance_Closed(q1,q2,Theta);

% Fnew = f2new;
iter = 1;
Hdiff = 100;

while (iter<itermax && Hdiff>0.0004)

    % Find phistar
    w = findphistarclosed(q2,Psi,b,Theta);

    v = q1 - q2; 

    % Find Update for Gamma
    gamupdate = findupdategamclosed(v,w,b,Theta);
    gamnew    = updategam(gamupdate,gamid,eps);

    gamp      = Apply_gam_gamid_closed(gamp,gamnew,n);
    
    % the Jacobian of gamp
    gamtmp = gamnew;
    gamtmp(:,:, 3) = zeros(size(gamnew, 1), size(gamnew, 2)); 
    [dgamp_du, dgamp_dv] = findgrad_closed(gamtmp, Theta);
    dgamp_du(:,:,3) = [];
    dgamp_dv(:,:,3) = [];
   
    J = zeros(size(gamnew, 1), size(gamnew, 2)); 
    for i=1:size(dgamp_du, 1),
        for j=1:size(dgamp_du, 2),
            J(i, j) =  dgamp_du(i, j, 1) * dgamp_dv(i,j, 2) - dgamp_du(i, j, 2) * dgamp_dv(i,j, 1);
        end
    end
    J = sqrt(abs(J));
%     imtool(J, []);
%     pause
%     
    % Update the surface   
%    Fnew = Apply_Gamma_Surf_Closed(Fnew,Theta,Phi,gamnew);
    q2 = Apply_Gamma_Surf_Closed(q2, Theta, Phi, gamnew);
    % q2 = q2 .* repmat(J, [1, 1, size(q2, 3)]);

    %iter
%     for j=1:3
%         Fnew(1,:,j)=Fnew(end,:,j);
%     end


    iter  = iter+1;


    %Calculate Distance
    H(iter) = Calculate_Distance_Closed(q1,q2,Theta);
    if (iter>1)
        if (H(iter)>H(iter-1))
            sprintf('ERROR: The step size is too large')
            break;
        end
    end

    if (iter>2)
        Hdiff = (H(iter-2)-H(iter-1))/H(iter-2);
    end
end

q2new = q2;
