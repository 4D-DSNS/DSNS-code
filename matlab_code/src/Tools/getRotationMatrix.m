function R = getRotationMatrix( axis, angle, is_radians)
%
% R = getRotationMatrix( axis, angle)
%
% Set up a 3x3 rotation matrix
%
% @return  rotation matrix
% @param  axis  rotation axis
% @param  angle  rotation angle
% @param  is_radians  if 1 then angle is in radians, otherwise it is in
%                     degrees

  % be sure the vector is normalized
  axis = axis / norm(axis);
  x = axis(1);  y = axis(2);  z = axis(3);

  ir = 0;
  if( nargin > 2)
    ir = is_radians;
  end
  is_radians = ir;
  
  % convert angle to radians
  if( 1 ~= is_radians)
    angle = deg_to_rad( angle);
  end

  % cache some values
  sin_a = sin( angle);
  cos_a = cos( angle);

  R = [cos_a + (1 - cos_a) * x^2 , ...
       (1 - cos_a) * x * y - sin_a * z, ...
       (1 - cos_a) * x * z + sin_a * y  ; ...
       (1 - cos_a) * x * y + sin_a * z, ...
       cos_a + (1 - cos_a) * y^2 , ...
       (1 - cos_a) * y * z - sin_a * x  ; ...
       (1 - cos_a) * x * z - sin_a * y, ...   
       (1 - cos_a) * y * z + sin_a * x, ...
       cos_a + (1 - cos_a) * z^2 ];
  
end