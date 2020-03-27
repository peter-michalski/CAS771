%this is for MNIST blurring
function expanded = encryptImages_MNIST(mydataset, numberIn, dimension)

%provide a 16 character key
key = 'dfrejf9834j3d89x';
keyAsNumber = 0;
for n = 1:16
    keyAsNumber = keyAsNumber + double(key(n));
end

while keyAsNumber >= 1.1
    keyAsNumber = sqrt(keyAsNumber);
end

multiplier = (2 - keyAsNumber);

expanded = zeros(numberIn,dimension,dimension);
for i = 1:numberIn
    grabbedImage = mydataset(:,:,i);
    for j = 1:dimension
        for k = 1:dimension
            element = grabbedImage(j,k);
            if (element < 0.2)
                expanded(i,j,k) = round((element + 0.47) * multiplier, 16);
            elseif ((element < 0.4) && (element >= 0.2))
                expanded(i,j,k) = round((element + 0.1) * multiplier, 16);
            elseif ((element < 0.6) && (element >= 0.4))
                expanded(i,j,k) = round(element * multiplier, 16);
            elseif ((element < 0.8) && (element >= 0.6))
                expanded(i,j,k) = round((element - 0.1) * multiplier, 16);
            elseif (element >= 0.8)
                expanded(i,j,k) = round((element - 0.47) * multiplier, 16);
            end    
        end
    end    
end