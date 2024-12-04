src1 = sprintf('%s%s',pwd,'/data/MeanShape/Spatiotemporal__00032_shortlong_bend_back_and_front/source/source.mat' );
target1 = sprintf('%s%s',pwd, '/data/MeanShape/Spatiotemporal__00127_shortlong_ATUsquat/target/reg.mat' );
target2 = sprintf('%s%s',pwd, '/data/MeanShape/Spatiotemporal__00145_shortlong_simple/target/reg.mat' );
target3 = sprintf('%s%s',pwd, '/data/MeanShape/Spatiotemporal__03331_longshort_simple/target/reg.mat' );
target4 = sprintf('%s%s',pwd, '/data/MeanShape/Spatiotemporal__00032_shortlong_bend_back_and_front/target/reg.mat' );
target5 = sprintf('%s%s',pwd, '/data/MeanShape/Spatiotemporal__00127_shortlong_bend_back_and_front/target/reg.mat' );
pca_basis_path = sprintf('%s%s',pwd, '/data/MeanShape/CAPE_PCA_256_256.mat' );

wd = pwd;
cd('matlab_code');
coregistration_mean({src1, target1, target2, target3, target4, target5}, pca_basis_path);
cd(wd);