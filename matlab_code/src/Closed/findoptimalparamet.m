function [f1,f2new,A2new,A_tmp12new,A_tmp22new,A_tmp21,optdodeca,optrot]=findoptimalparamet(f1,f2,Theta,Phi,scale)

[n,t,d]=size(f2);
O=DodecaElements;

[A1,A_tmp11,A_tmp21] = area_surf_closed(f1);

if scale == 1
   f1 = center_surface_closed(f1,A_tmp11,A1,Theta);
   f1 = scale_surf(f1,A1);
   [A1,A_tmp11,A_tmp21] = area_surf_closed(f1);
end

q1 = surface_to_q(f1,A_tmp21);
[A2,A_tmp12,A_tmp22] = area_surf_closed(f2);

if scale == 1
   f2 = center_surface_closed(f2,A_tmp12,A2,Theta);
   f2 = scale_surf(f2,A2);
end

for k=1:60
    for i=1:n
        for j=1:n
            [x(1),x(2),x(3)]=spherical_to_cart(Theta(i,j),Phi(i,j),1);
            y=O{k}*x';
            [Thetan(i,j,k),Phin(i,j,k),tmp]=cartesian_to_sph(y(1),y(2),y(3));
            if (Phin(i,j,k)<0)
                Phin(i,j,k)=Phin(i,j,k)+2*pi;
            end
        end
    end
    Thetan(:,:,k)=min(max(0,Thetan(:,:,k)),pi);
    Phin(:,:,k)=min(max(0,Phin(:,:,k)),2*pi);
end

for k=1:60
    gamnew{k}(:,:,1)=Thetan(:,:,k);
    gamnew{k}(:,:,2)=Phin(:,:,k);
    f2n{k}=Apply_Gamma_Surf_Closed(f2,Theta,Phi,gamnew{k});
    % for i=1:n
    %     for j=1:n
    %         f2n{k}(i,j,:)=O{k}'*squeeze(f2n{k}(i,j,:));
    %     end
    % end
    if scale == 1
       [A2n,A_tmp12n,A_tmp22n] = area_surf_closed(f2n{k});
       f2n{k} = center_surface_closed(f2n{k},A_tmp12n,A2n,Theta);
       f2n{k} = scale_surf(f2n{k},A2n);
    end

    [A2nn{k},A_tmp12nn{k},A_tmp22nn{k}] = area_surf_closed(f2n{k});

    if scale == 0
       A_tmp22n = A_tmp22nn{k};
    end

    q2n{k} = surface_to_q(f2n{k},A_tmp22n);
    distparam(k)=Calculate_Distance_Closed(q1,q2n{k},Theta);
end

[tmp,idx]=min(distparam);
H=distparam(idx);
f2new=f2n{idx};
A2new=A2nn{idx};
A_tmp12new=A_tmp12nn{idx};
A_tmp22new=A_tmp22nn{idx};
optdodeca=gamnew{idx};
optrot=O{idx};