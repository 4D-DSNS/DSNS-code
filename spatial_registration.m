
f1_src = sprintf('%s%s',pwd,'\data\Spatial_Registration\reference\' );
f2_src = sprintf('%s%s',pwd, '\data\Spatial_Registration\00127_shortlong_ATUsquat\' );

wd = pwd;
cd('matlab_code');
spatial_registration(f1_src,f2_src);
cd(wd);