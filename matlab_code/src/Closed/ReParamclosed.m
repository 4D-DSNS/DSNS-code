function [Fnew,H,gamp,iter1,A, combinedgamp] = ReParamclosed(f1,f2new,A_tmp21,A_tmp22new,b,gamp,gamid,Theta,Phi,Psi,iter1,A,winsize,itermax,eps)

[n,t,d]=size(f1);

q1 = surface_to_q(f1,A_tmp21);
q2 = surface_to_q(f2new,A_tmp22new);

H(1)=Calculate_Distance_Closed(q1,q2,Theta);

Fnew=f2new;
iter=1;
Hdiff = 100;

while (iter<itermax && Hdiff>0.0001)

    w = findphistarclosed(q2,Psi,b,Theta);
    v = q1 - q2; 

    % Find Update for Gamma
    nan_in_v = isnan(v);
    if any(nan_in_v)
        disp("Nan Value in V");
    end


    gamupdate = findupdategamclosed(v,w,b,Theta);
    gamnew = updategam(gamupdate,gamid,eps);

    combinedgamp{iter} = gamnew;

    gamp = Apply_gam_gamid_closed(gamp,gamnew,n);

    % Update the surface
    Fnew = Apply_Gamma_Surf_Closed(Fnew,Theta,Phi,gamnew); 

    %iter
    for j=1:3
        Fnew(1,:,j)=Fnew(end,:,j);
    end

    [Anew,multfactnew,sqrtmultfactnew] = area_surf_closed(Fnew);
    q2 = surface_to_q(Fnew,sqrtmultfactnew);

    iter  = iter+1;
    iter1 = iter1+1;

    %Calculate Distance
    H(iter) = Calculate_Distance_Closed(q1,q2,Theta);
    if (iter>1)
        if (H(iter)>H(iter-1))
            sprintf('ERROR: The step size is too large: Iteration: %d',iter)
            break;
        end
    end

    if (iter>2)
        Hdiff = (H(iter-2)-H(iter-1))/H(iter-2);
    end
end