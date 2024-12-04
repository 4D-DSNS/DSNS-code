function visualize4DSurfaces(XX, params)
% 
% XX - the 4D Geodesic
% params - parameters of the rendering
%          If params.isMean = 1 then the last 4D surfaceis the sample mean
%

if nargin < 2,
    params.delta = [.5, 0, 0];
    params.figId = 40;
    params.videofilename = []; 
    params.mycam = [0, 0, 30];     % For humans
    % params.mycam = [0, 0, 20];     % For faces
    params.isMean = 0;
    params.colorid = 22; 
    params.framePrefix = [];
    
    params.O = 0;
    
    params.faceColor1 = [.9  .9 .7];    % the normal shapes
    params.faceColor2 = [1  .9 .7];     % for the highlighted one (the mean)
end

if ~isfield(params, 'framePrefix')
    params.framePrefix = [];
end

if ~isfield(params, 'isMean')
    params.isMean = 0;
end

if ~isfield(params, 'mycam')
    params.mycam = [0, 0, 30];     % For humans
    % mycam = [0, 0, 20];     % For faces
end

if ~isfield(params, 'videofilename')
    params.videofilename = [];
end

if ~isfield(params, 'figId')
    params.figId = 40;
end

if ~isfield(params, 'delta')
    params.delta = [.5, 0, 0]; 
end

if ~isfield(params, 'O')
    params.O = 0;
end

if ~isfield(params, 'colorid')
    params.colorid = 22; 
end

if ~isfield(params, 'n_per_row')
    params.n_per_row = numel(XX);
end

if ~isfield(params, 'vis_mode')
    params.vis_mode = 0;        % flat - set it to 1 for faceted
end

nExpressions = numel(XX);  
nT           = size(XX{1}, 1);  % No. of temporal samples along each expression

f = figure(params.figId); %
f.Position = [10 10 1000 800];
clf;

if ~isempty(params.videofilename)
    v = VideoWriter(params.videofilename, 'MPEG-4');
    open(v);
end

colormap(bone(200));

d = params.delta; % * nExpressions/2.0; % [0, 0, 0]; 
D = linspace(-d(1)* params.n_per_row/2, d(1)* params.n_per_row/2, params.n_per_row)-0.32;

R = getRotationMatrix([0, 1, 0], params.O, 0);


[~, n,m,~] =size(XX{1});

if params.vis_mode == 1,
    % Colormap will be based on normals of a sphere   

    [Theta, Phi] = genGridSphr([n n], 0.01);
    [X, Y, Z]    = spherical_to_cart_m(Theta, Phi);
    C(:,:, 1) = 0.8 * ones(n,n); %X/2 + 0.5;
    C(:,:, 2) = .8* ones(n,n); %Y/2 + 0.5;
    C(:,:, 3) = .7* ones(n,n); %Z/2 + 0.5;
    
end


for t= 1:1:nT, % 113:347, %1:500, % nT   
    
%     if ((t>=134) && (t<=159)) || ((t>=367) && (t<=400))
%        continue; 
%     end
   
    
    clf; 
    axis off equal;    
    cameratoolbar;
    set(gca,'CameraViewAngleMode','manual');
    
    % x=(d(1)+1);
    % axis([ -x  x -.8  .8  -.75  .75 ]);
    % axis([ -x  x -1.2  1.2  -1.75  1.75 ]);
    axis([ D(1)-1  D(end)+1  -10  10  -10  10 ]);
   %  view(0, 90);
    campos(params.mycam); 
    camup([0, 1, 0]);
    camtarget([-.5,0,0])
    hold on;   
    
    %d(1) = -d(1)+.4;
       
    for i=1:nExpressions
        X = XX{i};
        M = rotate3D(squeeze(X(t, :, :, :)), R); % reconstructSurface(squeeze(Xgeod(i, :, t)), Mu, eigenVects, resolution);
        [h, w, ~] = size(M); 
        
        if (i == params.isMean) % nExpressions) && (params.isMean==1)
            theColors = params.faceColor2; % 180 * ones( h, w);  
            %continue
        else
            theColors = params.faceColor1; % 200*ones(h, w);%params.colorid * ones( h, w);
            dy = d(2);
        end
        
        if (i>1) && (i<nExpressions)
            dy = 0;%-.06;
        else
            dy = d(2);
        end        
        
        row_ix = floor((i-1) / params.n_per_row);
        col_ix = i-1 - row_ix * params.n_per_row;        
        
        dx = 0;
        if params.vis_mode == 0 % flat/smooth
            surface(squeeze(M(:, :, 1)) + D(1) + d(1) * col_ix, ... % + D(1) + dx + d(1) * col_ix, ... % D(i)* col_ix, ...
                    squeeze(M(:, :, 2)), ...%             + d(2) * row_ix, ... % dy   * row_ix, ...
                    squeeze(M(:, :, 3)), ...% +             d(3) * row_ix, ...
                    'FaceColor', theColors,...
                    'EdgeColor', 'none',...
                    'FaceLighting', 'gouraud',     ...
                    'AmbientStrength', 0.3, ...
                    'FaceAlpha',1);
        else    % faceted
            theColors = C;
            surf(squeeze(M(:, :, 1)) + D(1) + d(1) * col_ix, ... %+ D(1) + dx + d(1) * col_ix, ... % D(i)* col_ix, ...
                    squeeze(M(:, :, 2)), ...% -0.3 +         d(2) * row_ix, ... % dy   * row_ix, ...
                    squeeze(M(:, :, 3)), ...% +    d(3) * row_ix, ...
                    theColors); %, ...
%                     'EdgeColor', 'none',...
%                     'FaceLighting', 'gouraud',     ...
%                     'AmbientStrength', 0.3, ...
%                     'FaceAlpha',1);
            
        end


%         surface(squeeze(M(:, :, 1)) + d(1), squeeze(M(:, :, 2)) + d(2), squeeze(M(:, :, 3)) + d(3), ...
%                      theColors, ...
%                     'SpecularExponent', 100, ...
%                     'FaceAlpha', 1, ...                    
%                     'EdgeColor', 'none', ...
%                     'CDataMapping', 'direct');
                
        %d = d + params.delta;
    end
    
    %% Let's display now the mean
%     theColors = params.faceColor2;
%     i = params.isMean;
%     
%     if i>0,
%         row_ix = 0; % floor((i-1) / params.n_per_row);
%         col_ix = params.n_per_row-1;   
% 
%         X = XX{i};
%         M = rotate3D(squeeze(X(t, :, :, :)), R); % reconstructSurface(squeeze(Xgeod(i, :, t)), Mu, eigenVects, resolution);
%         [h, w, ~] = size(M); 
%         sc = 2; 
% 
%         if params.vis_mode == 0 % flat/smooth
%             surface(sc*squeeze(M(:, :, 1)) + D(1) + d(1) * col_ix + 1, ... % D(i)* col_ix, ...
%                     sc*squeeze(M(:, :, 2)), ...%  + D(end)/2 + d(2) * row_ix, ... % dy   * row_ix, ...
%                     sc*squeeze(M(:, :, 3)) -0.4, ...% + D(3) - d(3) * row_ix, ...
%                     'FaceColor', theColors,...
%                     'EdgeColor', 'none',...
%                     'FaceLighting', 'gouraud',     ...
%                     'AmbientStrength', 0.3, ...
%                     'FaceAlpha',1);
% 
%         else        
%             surface(sc*squeeze(M(:, :, 1)) + D(1) + d(1) * col_ix + 1, ... % D(i)* col_ix, ...
%                     sc*squeeze(M(:, :, 2)) , ... %+ D(end)/4, ...%  + d(2) * row_ix, ... % dy   * row_ix, ...
%                     sc*squeeze(M(:, :, 3)) -0.4);
%             shading faceted;        
% 
%         end
%     end
    
    h = camlight('left');
   % shading flat;
   %light; lighting phong;   
   % h = camlight('left');
    % axis equal; 
    
    
    frame = getframe(gcf);
    if ~isempty(params.videofilename)
        writeVideo(v,frame);
    end
    
    if ~isempty(params.framePrefix) && (mod(t, 1) == 0)
        % save one frame
        saveas(gcf, [params.framePrefix num2str(t)],'epsc');
    end
    
    pause(0.01);

end

if ~isempty(params.videofilename)
    close(v);
end



