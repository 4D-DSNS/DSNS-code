function [Theta,Phi] = genGridSphr(res,mysmall)

if (nargin < 2)
    mysmall = eps;
end

theta = linspace(mysmall, pi-mysmall,   res(2)); 
phi   = linspace(mysmall, 2*pi-mysmall, res(1)); 
Theta = repmat(theta, res(1),1);
Phi   = repmat(phi',1, res(2));

end


