function gamn = formdiffeosph(a,ShowDiffeo,n)

if (a<-1/pi || a>1/pi)
    sprintf('INVALID a')
end
b=0;
if (b<-1/(2*pi) || b>1/(2*pi))
    sprintf('INVALID b')
end

n=n-1;
theta = pi*[.01:n+1-.01]/(n+.02);
phi   = 2*pi*[0:n]/n;

theta = theta + a.*(theta-pi/(n+1)).*(theta-n*pi/(n+1));
phi   = phi   + b.*phi.*(phi-2*pi);
Theta = repmat(theta,n+1,1);
Phi   = repmat(phi',1,n+1);

gamn(:,:,1) = Theta;
gamn(:,:,2) = Phi;

if (ShowDiffeo==1)
    DisplayGrid_Closed(gamn,90);
end