function [theta,phi,r]=cartesian_to_sph(x,y,z)

r=sqrt(x^2+y^2+z^2);
theta=acos(z/sqrt(x^2+y^2+z^2));
% phi=atan2(y,x);
phi=atan2(x,y);

if phi<0
    phi=phi+2*pi;
end