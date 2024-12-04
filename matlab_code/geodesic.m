function geodesic(f1_src, f2_src, pca_basis_path, folderName)

addpath('src');
addpath('src\Closed');
addpath('src\Tools');

surfaces = {f1_src, f2_src};
PCAbasis = load(pca_basis_path);
delta_geodvis = [.5, 0, 0];

for i=1:numel(surfaces)
    F = load(surfaces{i}).S;
    [filepath, ~, ~] = fileparts(surfaces{i});
    nsample = size(F, 1);
    resolution = [size(F, 2), size(F, 3)];
    rotation = load([filepath,'/rotation.mat']).tmpoptrot;
    reparam = MakeClosedGrid(load([filepath, '/reparam.mat']).gamcum, resolution(1));

    S{i} = reparameterize(F, rotation, reparam, nsample, resolution);
end

M1 = projectExpressionToLowDim(S{1}, PCAbasis.Mu, PCAbasis.eigenVects); % 
M2 = projectExpressionToLowDim(S{2}, PCAbasis.Mu, PCAbasis.eigenVects); % 

doNormalizeScale = 0;
nGeodics = 5;
geodesic    = computeGeodesic(M1, M2, doNormalizeScale, nGeodics);

geod = cell(1, 1);
for i=1:nGeodics
    geod{i} = reconstructExpression(squeeze(geodesic(i, :, :)), PCAbasis.Mu, PCAbasis.eigenVects, resolution); 
end

[filepath,~,~] = fileparts(filepath);
[filepath,~,~] = fileparts(filepath);
videoOutDir = filepath;
dataSetName = 'geodesics';
params.isMean = 3;
params.delta  = delta_geodvis;
params.mycam = [0, 0, 20];
params.faceColor1 = [.9  .9 .7]; 
params.faceColor2 = [1  .9 .7];   
params.O = 45; 
params.videofilename = [ videoOutDir '/' dataSetName '_' folderName];
params.framePrefix = '';
% visualization
visualize4DSurfaces(geod , params);