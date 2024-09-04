%Project 2 Megan Lehmann

%filename = input("Enter file name: ",'s');
filename = 'Testimage1.tif';
im0 = imread(filename);
figure
imshow(im0);
title("Input Image");

%blur image
bk = (1/25)*ones(5);
im1 = imfilter(im0,bk);

%binarize image
im2=imbinarize(im1);

figure
imshow(im2);

%determine angle to rotate image
z = regionprops(im2,'Area','Centroid','BoundingBox','Orientation');
a = [z.Area];
maxArea = max(a);

[rCount,~] = size(z);

for i = 1:rCount
    if z(i).Area == maxArea
        regionIndex = i;
    end
end

rotation = z(regionIndex).Orientation;

precropped = imrotate(im0, -rotation); 
im3 = imrotate(im2,-rotation);  %black&white

%determine bounding box to crop to
p = regionprops(im3,'Area','Centroid','BoundingBox','Orientation');
a = [p.Area];
maxArea = max(a);

[rCount,~] = size(p);

for i = 1:rCount
    if p(i).Area == maxArea
        regionIndex = i;
    end
end

cropped = imcrop(precropped,p(regionIndex).BoundingBox);

[x,y] = size(cropped);

%rotate image 90 degrees if not oriented correctly
if (y>x)
    final = imrotate(cropped,90);
else
    final = cropped;
end
figure
imshow(final);
title("Output Image");