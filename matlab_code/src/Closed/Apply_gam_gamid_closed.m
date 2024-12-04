function gamcum = Apply_gam_gamid_closed(gamid,gaminc,n, Theta, Phi)

n  = n-1;

th0 = 0.01;
if nargin > 3,
    th0 = min(Theta(:));
end

ph0 = 0;
if nargin > 4,
    ph0 = min(Phi(:));
end

th = pi*[th0:n+1-th0]/(n+2*th0);
ph = 2*pi*[ph0: n-ph0]'/(n +2*ph0);

gamcum = zeros(size(gamid));

for i=1:n+1
    gamcum(i,:,1) = spline(th,gamid(i,:,1),gaminc(i,:,1));
end

for j=1:n+1
    gamcum(:,j,2) = spline(ph,gamid(:,j,2),gaminc(:,j,2));
end

