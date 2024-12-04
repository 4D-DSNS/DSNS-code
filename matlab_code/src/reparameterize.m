function S = reparameterize(S, rotation, reparam, nsample, resolution)

    [Theta,Phi] = genGridSphr(resolution);
    for k=1:nsample
        n = resolution(1);
        for i=1:n
            for j=1:n
                [x(1),x(2),x(3)]=spherical_to_cart(Theta(i,j),Phi(i,j),1);
                y=rotation*x';
                
                [Thetan(i,j),Phin(i,j),tmp]=cartesian_to_sph(y(1),y(2),y(3));
                if (Phin(i,j)<0)
                    Phin(i,j)=Phin(i,j)+2*pi;
                end
            end
        end
        Thetan(:,:)=min(max(0,Thetan(:,:)),pi);
        Phin(:,:)=min(max(0,Phin(:,:)),2*pi);
        
        gamnew(:,:,1)=Thetan(:,:);
        gamnew(:,:,2)=Phin(:,:);
        
        S(k,:,:,:)=Apply_Gamma_Surf_Closed(squeeze(S(k,:,:,:)),Theta,Phi,gamnew);
    
        S(k,:,:,:) = Apply_Gamma_Surf_Closed(squeeze(S(k,:,:,:)), Theta, Phi, reparam);
    end