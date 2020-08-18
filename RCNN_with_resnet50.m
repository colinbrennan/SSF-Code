
%%
clear
clc
close all

% load in training data as a groundTruth

unzip kelp_boat_lines.zip

image_data = load('kelp_boat_lines_groundTruth.mat')

image_data = image_data.gTruth

%%
image_dataset = objectDetectorTrainingData(image_data)

%%
height = height(image_dataset)

%%

%divide data into test, valdiation, and training. Populate into tables
rng(0)
shuffled_indicies = randperm(height);
indx = floor(0.6* height);

training_data_indx = 1:indx
training_data_table = image_dataset(shuffled_indicies(training_data_indx),:)

validation_data_indx = indx + 1 : indx + 1 +floor(.1*length(shuffled_indicies))
validation_data_table = image_dataset(shuffled_indicies(validation_data_indx),:)

test_indx = training_data_indx(end)+1 : length(shuffled_indicies);
test_data_table = image_dataset(shuffled_indicies(test_indx),:)

%%
%save data as image datastores and box label datastores
imds_train = imageDatastore(training_data_table.imageFilename)
blds_train = boxLabelDatastore(training_data_table(:,2:end))

imds_validation = imageDatastore(validation_data_table.imageFilename)
blds_validation = boxLabelDatastore(validation_data_table(:,2:end))

imds_test = imageDatastore(test_data_table.imageFilename)
blds_test = boxLabelDatastore(test_data_table(:,2:end))

%%
%combine corresponding box label datastores with their image datastores

training_data = combine(imds_train, blds_train);
validation_data = combine(imds_validation, blds_validation);
test_data = combine(imds_test, blds_test);

%%
data = read(training_data);

inputSize = [224 224 3]

% resize data to specified input size

preprocessedTrainingData = transform(training_data, @(data)preprocessData(data,inputSize));

%%
%define architecture

network = resnet50

featureLayer = 'activation_40_relu'

numClasses = 3

%%
% mirror and warp training data to add to it

augmentedTrainingData = transform(training_data,@augmentData);

trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
transform(validation_data, @(data)preprocessData(data,inputSize));

data = read(trainingData)

%%
%define training options
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 1, ...
        'InitialLearnRate', .001, ...
        'MaxEpochs', 3, ... 
        'Verbose', true)
%%
%train RCNN object detector or load in pre-trained network.
do_rcnn_training = false
   
if do_rcnn_training
   rcnn = trainRCNNObjectDetector(training_data_table, 'resnet50', options, 'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange', [.6 1]);
else
        load('rcnn.mat','rcnn')
   end
   

%%  

%visually test the classifier

testImage = imread('190516_125043_1.jpg')     
[bboxes,score,label] = detect(rcnn, testImage,'MiniBatchSize',128)
clear label_string
clear pos_data

for ii = 1:length(label)
    if  score(ii) >= .9
    label_string{ii} = sprintf('%s: (Confidence = %f)', label(ii,1), score(ii,1))
    pos_data(ii,:) = bboxes(ii,:)
    else
        label_string{ii} = sprintf('%s: (Confidence = %f)', label(ii,1), score(ii,1))
        pos_data(ii,:) = [zeros]
    end
    
end

outputImage = insertObjectAnnotation(testImage, 'rectangle', pos_data, label_string, 'FontSize',36, 'LineWidth',8);
figure
imshow(outputImage) 

%%
% set up and populate table for test data results

dims = size(test_data_table)
height = dims(1,1)
test_image_datastore = imageDatastore(test_data_table.imageFilename)

results = table('Size',[height 3], 'VariableTypes', {'cell','cell','cell'},'VariableNames',{'bboxes','scores','category'})

for i = 1:height
    I = readimage(test_image_datastore, i);
    [bboxes, score,label] = detect(rcnn, I,'MiniBatchSize',128);
      
    results.bboxes{i} = bboxes
    results.scores{i} = score
    results.category{i} = label
end


%%
%run test set through the detector

ds = imageDatastore(test_data_table.imageFilename)

detectionResults = detect(rcnn, ds.Files)


%test the accuracy
[ap,recall,precision]=evaluateDetectionPrecision(results, test_data_table(:,2:end), .5)

a = precision{3,1}
b = recall{3,1}

ap(3,1)

%Plot the accuracy vs recall
figure 
plot(b,a)
grid on
xlabel('Recall', 'fontsize', 12)
ylabel('Precision', 'fontsize', 12)
title(sprintf('Average precision = %.1f', ap(3,1)), 'fontsize', 18)
plotconfusion(results, test_data_table(:,2:end))
%%

function data = augmentData(data)
% Randomly flip images and bounding boxes horizontally.
tform = randomAffine2d('XReflection',true);
rout = affineOutputView(size(data{1}),tform);
data{1} = imwarp(data{1},tform,'OutputView',rout);
data{2} = bboxwarp(data{2},tform,rout);
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
data{2} = bboxresize(data{2},scale);
end



