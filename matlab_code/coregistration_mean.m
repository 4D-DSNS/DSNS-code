function coregistration_mean(surfaces_path, pca_basis_path)

    PCAbasis = load([pca_basis_path] );
    num_surfaces = numel(surfaces_path);
    delta_geodvis = [.5, 0, 0];

    for i=1:numel(surfaces_path)
        F = load(surfaces_path{i}).S;
        nsample = size(F, 1);
        resolution = [size(F, 2), size(F, 3)];
        [filepath, ~, ~] = fileparts(surfaces_path{i});
        rotation = load([filepath,'/rotation.mat']).tmpoptrot;
        reparam = MakeClosedGrid(load([filepath, '/reparam.mat']).gamcum, resolution(1));

        S{i} = reparameterize(F, rotation, reparam, nsample, resolution);
        M{i} = projectExpressionToLowDim(S{i}, PCAbasis.Mu, PCAbasis.eigenVects);
        q{i} = curve_to_q(M{i}, 0);
    end

    Mean_M = (1/num_surfaces).*(M{1}(:,1) + M{2}(:,1) + M{3}(:,1) + M{4}(:,1) + M{5}(:,1) + M{6}(:,1));
    Mean_q = (1/num_surfaces).*(q{1} + q{2} + q{3} + q{4} + q{5} + q{6});

    Xgeod_mean = zeros(1, size(q{1}, 1), size(q{1}, 2));
    Xgeod_mean(1,:,:) = q_to_curve(Mean_q, Mean_M);
    geod = cell(1, 1);

    for i=1:numel(S)
        geod{i} = S{i};
    end

    geod{7} = reconstructExpression(squeeze(Xgeod_mean(1, :, :)), PCAbasis.Mu, PCAbasis.eigenVects, resolution); 

    % visualization params
    [filepath,~,~] = fileparts(filepath);
    [filepath,~,~] = fileparts(filepath);
    videoOutDir = filepath;
    dataSetName = 'Coregistration';
    params.isMean = 7;
    params.delta  = delta_geodvis;
    params.mycam = [0, 0, 20];
    params.faceColor1 = [.9  .9 .7];    % the normal shapes
    params.faceColor2 = [1  .9 .7];     % for the highlighted one (the mean)
    params.O = 35;  
    params.videofilename = [ videoOutDir '/' dataSetName];
    params.framePrefix = '';
    % visualization
    visualize4DSurfaces(geod , params);