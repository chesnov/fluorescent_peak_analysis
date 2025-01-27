function hF = visualize_comps(S, sz)
    nS = size(S,2);
    RGB = rand(3,nS).^2; RGB = RGB./repmat(sum(RGB,1), 3,1);
    S_RGB = sqrt([S*RGB(1,:)' S*RGB(2,:)' S*RGB(3,:)']);
    S_RGB = 1.5* S_RGB./max(S_RGB(:));
    S_RGB = reshape(full(S_RGB), [sz 3]);
    hF = figure('Name', 'NMF components'); imshow(S_RGB);
end