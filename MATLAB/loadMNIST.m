function imagesOut = loadMNIST(filename)
%loadMNIST returns a 28x28xnumImages matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magicNum = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magicNum == 2051, ['Bad magicNum in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

imagesOut = fread(fp, inf, 'unsigned char');
imagesOut = permute(imagesOut,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
% Convert to double and rescale to [0,1]
imagesOut = double(imagesOut) / 255;

imagesOut = reshape(imagesOut, 28, 28, numImages);

end