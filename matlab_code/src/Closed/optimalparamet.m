function [f2new]=optimalparamet(f2,Theta,Phi, O, scale)

    [n,t,d]=size(f2);

    [A2,A_tmp12,A_tmp22] = area_surf_closed(f2);
    
    if scale == 1
       f2 = center_surface_closed(f2,A_tmp12,A2,Theta);
       f2 = scale_surf(f2,A2);
    end

    for i=1:n
        for j=1:n
            [x(1),x(2),x(3)]=spherical_to_cart(Theta(i,j),Phi(i,j),1);
            y=O*x';
            
            [Thetan(i,j),Phin(i,j),tmp]=cartesian_to_sph(y(1),y(2),y(3));
            if (Phin(i,j)<0)
                Phin(i,j)=Phin(i,j)+2*pi;
            end


            % if i==50 && j==100
            %     disp(O);
            %     disp(x');
            %     disp(y);
            %     disp(Phin(i,j));
            %     pause;

            % end

        end
    end
    Thetan(:,:)=min(max(0,Thetan(:,:)),pi);
    Phin(:,:)=min(max(0,Phin(:,:)),2*pi);

    gamnew(:,:,1)=Thetan(:,:);
    gamnew(:,:,2)=Phin(:,:);

    disp(gamnew(1:5,1:5,:));
    f2new=Apply_Gamma_Surf_Closed(f2,Theta,Phi,gamnew);
