f1_src = sprintf('%s%s',pwd,'\data\\Geodesics\\FaceTalk_170908_03277_TA_bareteeth\\S1.mat' );
f2_src = sprintf('%s%s',pwd, '\data\\Geodesics\\FaceTalk_170912_03278_TA_bareteeth\\S2.mat' );
pca_basis_path = sprintf('%s%s',pwd, '\data\\Geodesics\\COMA_PCA_129_129.mat' );

wd = pwd;
cd('matlab_code');
geodesic(f1_src,f2_src, pca_basis_path, 'registered');
cd(wd);
