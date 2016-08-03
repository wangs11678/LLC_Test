% =========================================================================
% 测试结果
% =========================================================================

clear all; close all; clc;
% -------------------------------------------------------------------------
% parameter setting
pyramid = [1, 2, 4];                % spatial block structure for the SPM
knn = 5;                            % number of neighbors for local coding
c = 10;                             % regularization parameter for linear SVM in Liblinear package 

gridSpacing = 6;
patchSize = 16;
maxImSize = 300;
nrml_threshold = 1;

% -------------------------------------------------------------------------
% set path
addpath('Liblinear/matlab');        % we use Liblinear package
img_dir = 'image/flower';     % directory for the image database                             
addpath('sift');
% -------------------------------------------------------------------------
subfolders = dir(img_dir);
database = [];

database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.nclass = 0;

% -------------------------------------------------------------------------
% load the codebook
Bpath = ['dictionary/Caltech101_SIFT_Kmeans_1024.mat'];
load(Bpath);
nCodebook = size(B, 2);              % size of the codebook  

% -------------------------------------------------------------------------
% load model
load('model.mat')
% -------------------------------------------------------------------------

kk = 1;
C = [];
for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') && ~strcmp(subname, '..'),
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        frames = dir(fullfile(img_dir, subname, '*.jpg'));
        
        c_num = length(frames);           
        database.imnum = database.imnum + c_num;
        database.label = [database.label; ones(c_num, 1)*database.nclass];
        
        for jj = 1:c_num,
            % -------------------------------------------------------------
            % extract SIFT descriptors
            disp('Extracting SIFT features...');
            imgpath = fullfile(img_dir, subname, frames(jj).name);
            
            I = imread(imgpath);
            if ndims(I) == 3,
                I = im2double(rgb2gray(I));
            else
                I = im2double(I);
            end;
            
            [im_h, im_w] = size(I);
            
            if max(im_h, im_w) > maxImSize,
                I = imresize(I, maxImSize/max(im_h, im_w), 'bicubic');
                [im_h, im_w] = size(I);
            end;
            
            % make grid sampling SIFT descriptors
            remX = mod(im_w-patchSize,gridSpacing);
            offsetX = floor(remX/2)+1;
            remY = mod(im_h-patchSize,gridSpacing);
            offsetY = floor(remY/2)+1;
    
            [gridX,gridY] = meshgrid(offsetX:gridSpacing:im_w-patchSize+1, offsetY:gridSpacing:im_h-patchSize+1);

            fprintf('Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches\n', ...
                     frames(jj).name, im_w, im_h, size(gridX, 2), size(gridX, 1), numel(gridX));

            % find SIFT descriptors
            siftArr = sp_find_sift_grid(I, gridX, gridY, patchSize, 0.8);
            [siftArr, siftlen] = sp_normalize_sift(siftArr, nrml_threshold);
            
            feaSet.feaArr = siftArr';
            feaSet.x = gridX(:) + patchSize/2 - 0.5;
            feaSet.y = gridY(:) + patchSize/2 - 0.5;
            feaSet.width = im_w;
            feaSet.height = im_h;

            % -------------------------------------------------------------
            % extract LLC features
            disp('Extracting LLC features...');
            dFea = sum(nCodebook*pyramid.^2);

            fea = LLC_pooling(feaSet, B, pyramid, knn);
            
            % -------------------------------------------------------------
            % evaluate the performance of the image feature using linear SVM
            % we used Liblinear package in this example code
            fprintf('Testing...\n');
            label = database.label(kk);
            [curr_C, ~, score] = predict(label, sparse(fea)', model);  
            
            %向服务器发送数据
            sendData(16, 1, curr_C, 'pend');
            
            C = [C; curr_C];
             
            fprintf('groundtruth:%d, predLable:%d\n', database.label(kk), C(kk));
            
            imgpath = retr_database_img_dir(img_dir);
            path = imgpath.path{kk};
            
            Image = imread(imgpath.path{kk});
            imshow(Image);
            %title('花卉识别');
            xlabel(['groundtruth = ',num2str(database.label(kk)),...
                  '  predLabel = ',num2str(C(kk))],...
                    'FontSize',14);            
            pause(2);

            if database.label(kk) ~= C(kk)
                disp('预测错误！');
                pause(10);
            end
            kk = kk + 1;
        end
    end
end

pred = 0;
for i = 1:database.imnum
    pred = pred + length(find(database.label(i) == C(i)));
end

fprintf('===============================================\n');
fprintf('Accuracy: %f(%d/%d)\n', double(pred/database.imnum),pred,database.imnum); 
fprintf('===============================================\n');
close all;