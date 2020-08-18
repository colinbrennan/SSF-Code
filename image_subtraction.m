
% filter out only upward-facing images

filter_up = dir('*_1.jpg*')

% define a reference

reference = imread('200123_182401_1.jpg');
reference_gray = rgb2gray(reference);

filter_up_cell = struct2cell(filter_up);


i = 1

A = []

while i<length(filter_up_cell);   
%load in test shot
test = imread(filter_up_cell{1,i});
test_gray = rgb2gray(test);

%compare
difference = imsubtract(reference_gray, test_gray);

%filter by intensity
limit = difference > 55;

%filter by area
filtered_image = bwareaopen(limit, 200000);

%filter by length
lengths = regionprops(filtered_image, 'MajorAxisLength');
Lengths = [lengths.MajorAxisLength];
index = Lengths > 20;
final_test = lengths(index);

if isempty(final_test);
  
else
 A = [A; string(filter_up_cell{1,i})]
    
end
i = i+1;
end










