%% 
clc,clear,close;
%% 模型预设超参数和常量The model presupposes hyperparameters and constants
[b1,a1] = butter(4, 0.02);
[b2,a2] = butter(4, 0.08, 'high');

T = 6000; % 总共6000张图像
T2 = 750; % 下采样后750张图像
%% 计算 IM - 读取图像
IM = zeros(128,128,6000);
video = VideoReader('CTZ2.avi');
i = 0;
% 使用第一个帧作为基准帧
fixedFrame = [];
while hasFrame(video)
    i = i + 1;
    frame = readFrame(video);
    IM(:,:,i) = rgb2gray(frame);
end
%% 计算 IMds - IMds是IM图像的副本，IMds对IM的通道维度下采样8倍 128 x 128 x 6000 -> 128 x 128 x 750
IMds = IM;
IMds = double(IM);
IMds = IMds(:,:,1:2:2*floor(end/2)) + IMds(:,:,2:2:end);
IMds = IMds(:,:,1:2:2*floor(end/2)) + IMds(:,:,2:2:end);
IMds = IMds(:,:,1:2:2*floor(end/2)) + IMds(:,:,2:2:end);
IMds = IMds./8;
%% 计算 F0 dF0 这一步骤占用大量内存，32G电脑会卡
%% Calculate F0 and dF0. This step takes up a lot of memory, 32GB computer will be slow.
% F0 - F0是IM滤波后结果平滑版IM
% dF0 - dF0是IM - F0，即过滤掉的部分
IM = double(IM);
IMds = double(IMds);
e1 =  medfilt2(reshape(IMds, [], size(IMds,3)), [1 3], 'symmetric');
v = 0;
a = 0.04;
for t = 2:size(e1,2)
    e1(:,t) = min(e1(:,t), e1(:,t-1) + v + a);
    v = max(0, e1(:,t) - e1(:,t-1));
end

F0ds = filtfilt(b1,a1,e1');
for ii = 1:6
    delta = min(0,e1'-F0ds);
    delta([1:20, end-19:end],:) = 0;
    F0ds = filtfilt(b1,a1,F0ds+2*delta); %2*delta to accelerate convergence
end
F0ds = reshape(F0ds', size(IMds));
F0 = reshape(interp1((0.5:1:T2).*(2.^3), reshape(F0ds,[],T2)',  1:T, 'linear', 'extrap')', [128, 128, 6000]);
dF = IM - F0; 
F0 = F0 - min(0, min(F0,[],3));
%% 计算  相关图 - F0ds 是通道下采样版本的F0，IMds是通道下采样版本的IM，用通道下采样的F0ds计算相关图
%%Calculate the correlation graph - F0ds is the F0 of the subsampled version of the channel. IMds is the IM of the subsampled version of the channel. Calculate the correlation graph with the F0ds of the subsampled channel
dFds = IMds - F0ds;
dFdshp = permute(filtfilt(b2,a2,permute(dFds, [3 1 2])), [2 3 1]);
disp('computing correlation image');
ss = sum(dFdshp.^2,3);
vertC = sum(dFdshp .* circshift(dFdshp, [1 0 0]),3)./sqrt(ss.*circshift(ss, [1 0 0]));
horzC = sum(dFdshp .* circshift(dFdshp, [0 1 0]),3)./sqrt(ss.*circshift(ss, [0 1 0]));
C = nanmean(cat(3, horzC, circshift(horzC,1,2), vertC, circshift(vertC, 1,1)),3);
%% 寻找C图像峰值 - 粗筛
%% Look for C-image peaks - rough screening
sigma = 2;
bgFrac = 0.2;
maxN = 600;
nIter = 5;
sparseFac = 0.15;

sz = [128, 128];
C = imgaussfilt(C,0.5);
C2 = C;
Cthresh = 2.*(median(C(:))-prctile(C(:),1));
C2(C2<(median(C(:))+Cthresh)) = 0;
BW = imregionalmax(C2);
C2(imdilate(BW, strel('disk', 6*sigma))) = 0;
while any(C2(:))
   BW = BW | imregionalmax(C2);
   C2(imdilate(BW, strel('disk', 6*sigma))) = 0;
end
BWinds = find(BW(:));
if length(BWinds)>maxN
    BWvals = C(BWinds);
    [~,sortorder] = sort(BWvals, 'descend');
    BWinds = BWinds(sortorder(1:maxN));
end
nComp = length(BWinds);
[Pr,Pc] = ind2sub(sz, BWinds); %locations of putative release sites

nBG = floor(nComp.*bgFrac+10);
W0 = zeros([sz nComp+nBG]); %start with twice as many components as local maxima
W0(sub2ind(size(W0), Pr',Pc', 1:nComp)) = 1; %initialize one pixel for every component
W0 = imgaussfilt(W0,0.5)+imgaussfilt(W0,sigma, 'FilterSize', 6*ceil(sigma)+1);
W0 = reshape(W0, prod(sz),nComp+nBG);
W0(:, nComp+1:end) = rand(size(W0,1), size(W0,2)-length(Pr))./size(W0,1);

opts1 = statset('MaxIter', 20,  'Display', 'iter');%, 'UseParallel', true);
[W0,H0] = nnmf(reshape(dFds,[],T2), nComp+nBG,'algorithm', 'mult', 'w0', W0, 'options', opts1);
%% 寻找C图像峰值 - 细筛
%% Look for C-image peaks - careful screening
for bigIter = 1:nIter
    disp(['outer loop ' int2str(bigIter) ' of ' int2str(nIter)]);
    nW0 = sum(W0>0,1);
   
    %apply sparsity
    setZero = W0<(sparseFac.*max(W0,[],1));
    setZero(:, nW0<=9) = false; %don't shrink any more once below 9 pixels
    W0(setZero) = 0;
    
    %apply contiguous constraint
    smallComps = nW0<(prod(sz)*0.05); %sparse components, that we will apply contiguous constrains to; we have to do this because matlab's nnmf reorders components
    W0 = reshape(W0, sz(1),sz(2),[]);
    for comp = find(smallComps)
        [maxval, maxind] = max(reshape(W0(:,:,comp),1,[]));
        [rr,cc] = ind2sub(sz, maxind);
        if nW0(comp)<5
            W0(max(1,min(end,rr+(-1:1))),max(1,min(end,cc+(-1:1))),comp) = maxval/3;
        end
        W0(:,:,comp) = W0(:,:,comp).*bwselect(W0(:,:,comp)>0, cc,rr, 4);
    end
    W0 = reshape(W0, prod(sz),[]);
    [W0,H0] = nnmf(reshape(dFds,[],T2), nComp+nBG,'algorithm', 'mult', 'w0', W0, 'h0', H0, 'options', opts1);
end
%% 寻找C图像峰值 - 合并相同信号
%% Look for C-image peaks - merge the same signals
dF = reshape(dF,[],T);
F0 = reshape(F0,[],T);

Hhf = W0\dF;
nW0 = sum(W0>0,1);
smallComps = nW0<(prod(sz)*0.01);
recalc = false;
activityCorr = corr((Hhf-smoothdata(Hhf,2,'movmean',75))', 'type', 'Spearman');
for c1 = 1:size(W0,2)
    for c2 = (c1+1):size(W0,2)
        if all(smallComps([c1 c2])) && (activityCorr(c1,c2)>0.25) && any(imdilate(reshape(W0(:,c1),sz), ones(3)) & reshape(W0(:,c2),sz),'all')
            W0(:,c1) = W0(:,c1) + W0(:,c2);
            W0(:,c2) = 0;
            activityCorr(c2,:) = 0; activityCorr(:,c2) = 0;
            recalc = true;
        end
    end
end
if recalc
    disp('Some components were merged. Recalculating factorization.');
    sel = any(W0,1);
    W0 = W0(:, sel);
    H0 = H0(sel,:);
    %run some more NMF
    [W0,~] = nnmf(reshape(dFds,[],T2), sum(sel),'algorithm', 'mult', 'w0', W0, 'h0', H0, 'options', opts1);
    Hhf = W0\dF; %solve for full speed data
end
%% 寻找C图像峰值，可视化分割
%% Look for C-image peaks -visual segmentation 

hFvs = [];
nW0 = sum(W0>0,1);
smallComps = nW0<(prod(sz)*0.01); 
W0 = W0(:,smallComps);
Hhf = Hhf(smallComps,:);
delete(hFvs)
hFvs = visualize_comps(W0,sz);
set(hFvs, 'name', 'demo');
drawnow;

%% 信号分解，信号排序+
%% Signal decomposition and sorting
ac = sum(W0,1)'.*Hhf;
[~,sortorder] = sort(sum((ac-smoothdata(ac,2,'movmean',75)).^2,2), 'descend');
W0 = W0(:,sortorder);
Hhf = Hhf(sortorder,:);
DFF=nan(size(Hhf)); rawDFF = DFF;
F=nan(size(Hhf)); rawF = F; Fzero = F;
lambda = 10; %prctile(meanIM(meanIM(:)>0),1); %regularizer, larger favors selecting brighter pixels
for comp = 1:size(Hhf,1)
    support = find(W0(:,comp)>0);
    F(comp,:) = sum((W0(support,comp).*Hhf(comp,:)),1);
    pxDFF = (W0(support,comp).*Hhf(comp,:))./(F0(support,:)+lambda);
    [~, sortorder] = sort(sqrt(sum(pxDFF.^2,2)), 'descend');
    selpix = sortorder(1:min(end,9)); %take the 9 highest-DFF pixels
    Fzero(comp,:) = sum(F0(support(selpix),:),1);
    DFF(comp,:)= sum(W0(support(selpix),comp)*Hhf(comp,:),1)./Fzero(comp,:);
    rawF(comp,:) = sum(dF(support(selpix),:),1);
    rawDFF(comp,:) = sum(dF(support(selpix),:),1)./Fzero(comp,:);
end
%% 信号保存
%% Save
A = struct();
A.lambda = lambda;
A.IM = mean(IMds,3);
A.F0 = Fzero;
A.DFF = DFF; 
A.rawDFF = rawDFF;
A.F = F;
A.rawF = rawF;
A.spatial = reshape(W0, [sz size(W0,2)]);
save('CTZ3.mat', 'A');
dlmwrite('CTZ3.rawDFF.txt',A.rawDFF);
dlmwrite('CTZ3.DFF.txt',A.DFF);