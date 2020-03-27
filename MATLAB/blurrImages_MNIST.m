%this is for MNIST blurring
function expanded = blurrImages_MNIST(mydataset, numberIn, varianceIn)

expanded = zeros(numberIn,28,28);
for i = 1:numberIn
grabbedImage = mydataset(:,:,i);
blurredImage = imnoise(grabbedImage, 'gaussian', 0, varianceIn);
blurredImageOut = round(blurredImage, 16);
expanded(i,:,:) = blurredImageOut;
end