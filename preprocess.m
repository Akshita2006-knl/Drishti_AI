function preprocess()
% Automated Full-Stack Server Pipeline Hook


% Since the script is inside input_images, read sample_eye.jpg directly from here
    function preprocess()
        % Automated Full-Stack Server Pipeline Hook

        % Point explicitly to the input_images subfolder where Python saves the image
        inputFolder = 'input_images';
        inputFile = fullfile(inputFolder, 'sample_eye.jpg');

        % Point explicitly to the processed_images folder
        outputFolder = 'processed_images';
        outputFile = fullfile(outputFolder, 'cleaned_eye.png');

        fprintf('[INFO] Initializing image normalization pipeline...\n');


if ~exist(inputFile, 'file')
    error('[ERROR] "sample_eye.jpg" not found! Please place a sample image inside the input_images folder next to this script.');
end

rawImage = imread(inputFile);
if size(rawImage, 3) == 3
    greenChannel = rawImage(:, :, 2);
else
    greenChannel = rawImage;
end

enhancedImage = adapthisteq(greenChannel, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');

if size(rawImage, 3) == 3
    grayImage = rgb2gray(rawImage);
else
    grayImage = rawImage;
end
binaryMask = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.1);
stats = regionprops(binaryMask, 'BoundingBox', 'Area');

if ~isempty(stats)
    [~, maxIdx] = max([stats.Area]);
    croppedImage = imcrop(enhancedImage, stats(maxIdx).BoundingBox);
else
    croppedImage = enhancedImage;
end

targetDimension = [224, 224];
resizedImage = imresize(croppedImage, targetDimension);
finalRGB = cat(3, resizedImage, resizedImage, resizedImage);

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
imwrite(finalRGB, outputFile);

fprintf('[SUCCESS] Processed image saved perfectly to: %s\n', outputFile);
end