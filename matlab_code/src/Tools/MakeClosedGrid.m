function fn = MakeClosedGrid(f, n2, interp_type)

mysmall = 0.01;

if nargin < 3,
    interp_type = 'spline';
end

if size(n2) ==1,
    n2 = [n2 n2];
end

[n1,t1,d1] = size(f);

%% old resolution
[Theta1,Phi1] = genGridSphr([n1, n1],  mysmall);

%% new resolution
[Theta,Phi] = genGridSphr([n2(1), n2(2)], mysmall);

%% Interpolation
for i=1:d1,
    fn(:,:,i) = interp2(Theta1,Phi1,f(:,:,i),Theta,Phi,interp_type); % 'spline');
end

% fn(:,:,2) = interp2(Theta1,Phi1,f(:,:,2),Theta,Phi,'spline');
% fn(:,:,3) = interp2(Theta1,Phi1,f(:,:,3),Theta,Phi,'spline');


