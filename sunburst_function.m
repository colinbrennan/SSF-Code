clear
clc
close all

start = [1356,930,744,99];
figure
imshow(pixel_values(start));
figure


%ans = fminsearch(@compare, start);





%%
function summation = compare(m)
    
    test_image = imread('200123_171546_1.jpg');
    test_gray = rgb2gray(test_image);
    test_gray_double = double(test_gray);
    test_gray_mat = mat2gray(test_gray_double);
    m = pixel_values(m);
    dif = imsubtract(m, test_gray_mat);
    dif_2 = mat2gray(dif);
    summation = sum(sum(dif_2));
end
%%


function pv = pixel_values(p)
    x_o = p(1);
    y_o = p(2);
    r_1 = p(3);
    r_2 = p(4);
    x = (0:1:2591);
    y = (0:1:1943);
    y_T = transpose(y);
    r = sqrt(((x-x_o).^2)+(y_T-y_o).^2);
    pv_raw = (r-r_1)/(r_2-r_1);
    pv = pv_raw/(max(max(pv_raw)));
    
end


%%


