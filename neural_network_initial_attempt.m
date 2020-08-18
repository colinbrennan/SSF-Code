
%%
clear
clc
close all
%%
imds = (imageDatastore('/Users/colinbrennan/Desktop/test'))

%%
imageLabeler(imds);


%%
image_size = [1944 2592 3]

numClasses = 4

anchorBoxes = [1 1;4 6;5 3;9 6];

network = resnet50()

analyzeNetwork(network)

featureLayer = 'activation_49_relu';

%%
load(gTruth);
%%
data = pixelLabelImageDatastore(gTruth);


%%
layers = [imageInputLayer([1944 2592 3]),convolution2dLayer(12,25), reluLayer, maxPooling2dLayer(2, 'Name', 'maxPooling', 'stride',1,'Padding',[1,1]), fullyConnectedLayer(3), softmaxLayer, pixelClassificationLayer]
%analyzeNetwork(layers)
%%
training_set = objectDetectorTrainingData(gTruth);

%%
layers2 =  [imageInputLayer([1944 2592 3]),convolution2dLayer(3,16,'Stride',2,'Padding',1), reluLayer, maxPooling2dLayer(2,'Stride',2), fullyConnectedLayer(3),softmaxLayer, classificationLayer]
analyzeNetwork(layers2)
%%
options = trainingOptions('sgdm', 'MiniBatchSize', 10, 'InitialLearnRate', 1e-3, 'MaxEpochs', 20);
%%
layers3 = [imageInputLayer([1944 2592 3]), convolution2dLayer(1,3), reluLayer, maxPooling2dLayer(1,'Stride',1), convolution2dLayer(1,1), fullyConnectedLayer(5), softmaxLayer, classificationLayer]
analyzeNetwork(layers3)

%%
size(training
%%
detector = trainNetwork(data, layers3, options)

%%
[pred, scores] = classify(detector, imds);
%%
Ypred = detector('yes190516_124455_1.jpg');
%%
img = imread('yes190516_124455_1.jpg');

[bboxes,scores] = detect(acfDetector,img);
%%
for i = 1:length(scores)
   annotation = sprintf('Confidence = %.1f',scores(i));
   img = insertObjectAnnotation(img,'rectangle',bboxes(i,:),annotation);
end

figure
imshow(img)