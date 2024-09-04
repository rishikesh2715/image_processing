%Determine the file from user containing all images
directory = uigetdir('C:\');

%get all jpg files from given folder
myImages = dir(fullfile(directory, '*.jpg'));

for i = 1:length(myImages)
    %use nested for loops to identify RGB values, 
    % and use them to determine if each pixel is 'gray'
    im = imread(fullfile(directory,myImages(i).name));
    [r, c, chan] = size(im);
    Gr = 0;
    for a = 1:r
        for b = 1:c
            R = im(a,b,1);
            G = im(a,b,2);
            B = im(a,b,3);
            
            if (R == G) && (G == B)
                Gr = Gr + 1;
            end
        end
    end    
    %determine if there are more gray than colored pixels to determine if
    %the whole picture is day or night
    total = r*c;
    if total/Gr > 2
        str = "Day";
    else 
        str = "Night";
    end
    
    %add day/night text to the image
    im = insertText(im, [100 100], str,'FontSize',18,'BoxColor','w');
    
    %display the image, waiting for key presses after each one
    imshow(im)
    pause;
end