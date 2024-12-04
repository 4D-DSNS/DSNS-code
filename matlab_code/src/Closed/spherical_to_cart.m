function [x,y,z]=spherical_to_cart(theta,phi,r)

x=r*sin(theta)*sin(phi);
y=r*sin(theta)*cos(phi);
z=r*cos(theta);
