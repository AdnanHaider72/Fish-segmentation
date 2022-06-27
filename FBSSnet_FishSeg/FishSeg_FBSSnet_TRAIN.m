
close all
clear all

load 'FBSSSnet_lgraph' % load lgraph of EFS-Net 
lgraph=layers_2;

Folder ='';% Main directory to all images
 
train_img_dir = fullfile(Folder,'Train');%Training image directory
imds = imageDatastore(train_img_dir); 
Val_img_dir = fullfile(Folder,'Val');%Training image directory
imdsVal = imageDatastore(Val_img_dir); 

classes = ["FISH","BG"]; %% Class names
labelIDs   = [255,0]; % Class id


train_label_dir = fullfile(Folder,'Train_GT');  %% Training label directory
pxds = pixelLabelDatastore(train_label_dir,classes,labelIDs);

Val_label_dir = fullfile(Folder,'Val_GT');  %% Validation label directory
pxdsVal = pixelLabelDatastore(Val_label_dir,classes,labelIDs);

tbl = countEachLabel(pxds); % occurance of Fish and non-Fish pixels


frequency = tbl.PixelCount/sum(tbl.PixelCount); % frequency of each class

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;     


%%% Training options %%%%%

dsVal = combine(imdsVal,pxdsVal);

options = trainingOptions('adam', ...
    'SquaredGradientDecayFactor',0.95, ...
    'GradientThreshold',5, ...
    'GradientThresholdMethod','global-l2norm', ...
    'Epsilon',1e-6, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',130, ...  
    'MiniBatchSize',5, ...
    'CheckpointPath',tempdir, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',2, ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,35));
    

augment_data = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-5 5],'RandYTranslation',[-5 5]); % optional data augmentation


training_data = pixelLabelImageDatastore(imds,pxds,...
    'DataAugmentation',augment_data); 


[net, info] = trainNetwork(training_data,lgraph,options);% Train the network



function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end

  

