clear
clc 
close all
%%

%add CIFAR-10 help function to path

fullfile(matlabroot,'examples','deeplearning_shared','main','helperCIFAR10Data.m')
addpath(fullfile(matlabroot,'examples','deeplearning_shared','main'))

cifar10Data = cd;



%%

%load in training data

[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);



%%

%Begin defining layers

size = [32,32,3]

input_layer = imageInputLayer(size)

middle_layer = [convolution2dLayer([5 5],32,'Padding',2), reluLayer(),maxPooling2dLayer(3,'Stride',2),convolution2dLayer([5 5],32,'Padding',2),reluLayer(),maxPooling2dLayer(3,'Stride',2),convolution2dLayer([5 5],64,'Padding',2), reluLayer(),maxPooling2dLayer(3,'Stride',2)]

closing_layers = [fullyConnectedLayer(64),reluLayer, fullyConnectedLayer(10),softmaxLayer,classificationLayer], 
%%
Layers = [input_layer, middle_layer, closing_layers]
%analyzeNetwork(Layers)

%%
%define training options

opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true, 'Plots', 'training-progress');

%%
%either train netowrk or load in pre-trained network

doTraining = false;

if doTraining    
   
    network = trainNetwork(trainingImages, trainingLabels, Layers, opts);
    save network
else
     load('network.mat', 'network')
end

%%
%Determine the accuracy of initial network
YTest = classify(network, testImages);
accuracy = sum(YTest == testLabels)/numel(testLabels)


plotconfusion(testLabels, YTest)

%%
%load in Kelpcam training data as a GroundTruth
unzip kelp_boat_lines.zip

image_data = load('kelp_boat_lines_groundTruth.mat')

image_data = image_data.gTruth

%%
%Compile image data and pixel labels as asingle table
image_dataset = objectDetectorTrainingData(image_data)

%%

height = height(image_dataset)

%%
%shuffle data and divide into training, test, and validation sets

rng(0)
shuffled_indicies = randperm(height);
indx = floor(0.8* height);

training_data_indx = 1:indx
training_data_table = image_dataset(shuffled_indicies(training_data_indx),:)

test_indx = training_data_indx(end)+1 : length(shuffled_indicies);
test_data_table = image_dataset(shuffled_indicies(test_indx),:)

%%

%define options for RCNN detector

options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', .001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 50, ... 
        'Verbose', true);

%%
%train RCNN detector
do_training_rcnn = true;
  
 if do_training_rcnn   
    rcnn = trainRCNNObjectDetector(training_data_table, network, options, 'NegativeOverlapRange', [0 0.1], 'PositiveOverlapRange',[0.4 1])
    save rcnn
 else
     load('rcnn.mat','rcnn')
 end
 

%%
numImages = height(test_data_table)
%%
%set up a results table for test set trial
numImgaes = 14
results = table('Size',[14 6], 'VariableTypes', {'cell','cell','cell','cell','cell','cell'},'VariableNames',{'boat','Scores_1','kelp','Scores_2','lines','Scores_3'})

%%
test_image_datastore = imageDatastore(test_data_table.imageFilename)

%%

%output results intpo talbe
for i = 1:14
    I = readimage(test_image_datastore, i);
    [bboxes, score,label] = detect(rcnn, I,'MiniBatchSize',128);
  for ii = 1:length(score)
      
    if label(ii) == string('kelp')
        if score(ii) > .8
        results.kelp{i} = bboxes(ii,:);
        results.Scores_2{i} = score(ii);
        else
            results.kelp{i} = [];
            results.Scores_2{i} = [];
        end
    end
   
   
    if label(ii) == string('lines')
        if score(ii) > .8
        results.lines{i} = bboxes(ii,:);
        results.Scores_3{i} = score(ii);
        else
            results.lines{i} = [];
            results.Scores_3{i} = [];
        end
    end
    
    if label(index) == string('boat')
        if score(ii) > .8
        results.boat{i} = bboxes(ii,:);
        results.Scores_1{i} = score(ii);
        else
            results.boat{i} = [];
            results.Scores_1{i} = [];
        end
    end
  end
end
 
%%

%evaluate precision
[ap,recall,precision]=evaluateDetectionPrecision(results(:,3:4), test_data_table(:,3), .5)
figure 
plot(recall,precision)
grid on
  
    %%
      clear label_string
      clear pos_data

%%
%visually test
testImage = imread('190516_124401_2.jpg');
      
%%
      
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128)
  
 
%%

%output detection data into a string containing labels and an array
%containg location data

for ii = 1:length(label)
    if  score(ii) >= .7
    label_string{ii} = sprintf('%s: (Confidence = %f)', label(ii,1), score(ii,1))
    pos_data(ii,:) = bboxes(ii,:)
    else
        label_string{ii} = sprintf('%s: (Confidence = %f)', label(ii,1), score(ii,1))
        pos_data(ii,:) = [zeros]
    end
    
end


%%
%show result
outputImage = insertObjectAnnotation(testImage, 'rectangle', bboxes, label_string, 'FontSize',36, 'LineWidth',8);
figure
imshow(outputImage)  




