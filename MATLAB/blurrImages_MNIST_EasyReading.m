%this is for MNIST blurring
function expanded = blurrImages_MNIST_EasyReading(mydataset, numberIn, varianceIn)

expanded = zeros(28,28,numberIn);
for i = 1:numberIn
grabbedImage = mydataset(:,:,i);
blurredImage = imnoise(grabbedImage, 'gaussian', 0, varianceIn);
expanded(:,:,i) = blurredImage;
end