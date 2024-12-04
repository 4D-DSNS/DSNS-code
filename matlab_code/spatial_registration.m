function spatial_registration(F1_src, F2_src)

% configure;
wd = pwd;

F1 = double(load([F1_src, '/first_frame.mat']).S);
F2 = double(load([F2_src, '/first_frame.mat']).S);

cd('src/Closed');


n = size(F1,1);

A = 0;
winsize=[0,0,100,100];
mineps=1e-10;

[Theta,Phi,Psi,b] = spherharmbasis1(2,n);
gamid = MakeDiffeo_Closed(0,n,0,b);

rot_path = [sprintf('%s/rotation.mat', F2_src)];
reparam_path = [sprintf('%s/reparam.mat', F2_src)];

[A1,multfact1,~] = area_surf_closed(F1);
F1 = center_surface_closed(F1,multfact1,A1,Theta);
F1 = scale_surf(F1,A1);
[~,~,sqrtmultfact1] = area_surf_closed(F1);
q1 = surface_to_q(F1,sqrtmultfact1);


[A2,multfact2,sqrtmultfact2] = area_surf_closed(F2);
F2 = center_surface_closed(F2,multfact2,A2,Theta);
F2n = scale_surf(F2,A2);
[~,~,sqrtmultfact2n] = area_surf_closed(F2n);
q2n = surface_to_q(F2n,sqrtmultfact2n);

beforereg=Calculate_Distance_Closed(q1,q2n,Theta);

[f1,f2new,~,~,~,A_tmp21,~,tmpoptrot]=findoptimalparamet(F1,F2n,Theta,Phi, 0);

[~,~,A_tmp22new] = area_surf_closed(f2new);
q2n = surface_to_q(f2new,A_tmp22new);

eps=0.1;
run=1;

while run == 1
      
   [F2n,H1,gamcum,iter1,A, ~] = ReParamclosed(f1,f2new,A_tmp21,A_tmp22new,b,gamid,gamid,Theta,Phi,Psi,2,A,winsize,500,eps);
   
   run = 0;
   if eps < mineps
      fprintf('Giving up.\n');
      run = -1;
   else
      % eps = eps/2;
      fprintf('Trying again, eps = %g ...\n', eps);
   end
end
      
[~,~,sqrtmultfact2n] = area_surf_closed(F2n);
q2n = surface_to_q(F2n,sqrtmultfact2n);

run = 1;

while run == 1
         
   [F2n,H2,gamcum,iter1,A, ~] = ReParamclosed(f1,F2n,sqrtmultfact1,sqrtmultfact2n,b,gamcum,gamid,Theta,Phi,Psi,iter1+1,A,winsize,500,eps);
   run = 0;

   if eps < mineps
      fprintf('Giving up.\n');
      run = -1;
   else
      fprintf('Trying again, eps = %g ...\n', eps);
   end
end
      
[~,~,sqrtmultfact2n] = area_surf_closed(F2n);
q2n = surface_to_q(F2n,sqrtmultfact2n);

run = 1;

while run == 1
      
   [Fnew,H3,gamcum,iter1,A, ~] = ReParamclosed(f1,F2n,sqrtmultfact1,sqrtmultfact2n,b,gamcum,gamid,Theta,Phi,Psi,iter1+1,A,winsize,500,eps);
   run = 0;

   if eps < mineps
      fprintf('Giving up.\n');
      run = -1;
   else
      % eps = eps/2;
      fprintf('Trying again, eps = %g ...\n', eps);
   end
end

afterreg=Calculate_Distance_Closed(q1,q2n,Theta);
fprintf('Before Registration = %g, After Registration = %g\n', beforereg, afterreg);

% Preallocate F to the correct size
Funcsdion = zeros(n, n, 7, 3);  % Assuming F is nxnx7x3

f1_num = F1;
f2_num = Fnew;

for fseze=1:7
   s=(fseze-1)/6;
   Funcsdion(:,:,fseze,:)=f1_num.*(1-s)+f2_num.*s;
end
DisplayGeod(Funcsdion,5,n,n);


save(rot_path, 'tmpoptrot');
save(reparam_path, 'gamcum');


fprintf('Done! Saving to %s ...\n', F2_src);
cd(wd);