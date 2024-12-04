function H = Calculate_Distance_Closed(q1,q2,Theta)

tmp = q1-q2;
[a1,a2,a3] = size(tmp);
n=a1-1;

X = sum(tmp.*tmp,3).*sin(Theta);
H = sqrt(sum(X(:)) * ((n*pi+pi-.02*pi)/(n^2+.02*n)).*(2*pi/(n)));

% H = sqrt(sum(sum(sum(tmp.*tmp,3).*sin(Theta))).*((n*pi+pi-.02*pi)/(n^2+.02*n)).*(2*pi/(n)));