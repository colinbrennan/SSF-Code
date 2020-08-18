clear
clc
close all


% filtering out only images from upward-facing camera

filter_up = dir('*_1.jpg*')
filter_up = struct2cell(filter_up);
[r,c] = size(filter_up);
%%
img_one = char(filter_up(1,1));
test_image = imread(img_one);
[pixel_rows, pixel_columns, colorChannels] = size(test_image);

%%
  
% process all of the images
images_of_interest = []

for i = 1:c;   
    
%load in test shot
test = imread(filter_up{1,i});
test_gray = rgb2gray(test);

%%

%filter using edge detection
test_gray_edged = edge(test_gray, 'prewitt', .005, 'nothinning');


%filter returns by area
filtered_image = bwareaopen(test_gray_edged, 10000);


%filter returns by length
lengths = regionprops(filtered_image, 'MajorAxisLength');
lengths_row_vector = [lengths.MajorAxisLength];
index = lengths_row_vector > 100;
filtered_lengths = lengths_row_vector(index);

%%show returns in an array with pose/position data

if isempty(filtered_lengths);
  
else
 info = imfinfo(filter_up{1,i});
 images_of_interest = [images_of_interest; string(filter_up{1,i}), info.Comment];
    
end
end
%%



 








