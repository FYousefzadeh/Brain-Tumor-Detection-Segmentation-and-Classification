clc;
clear;
close all;

%% Code

%% Training Data
registry = imformats; %get all image type supported by matlab
extensions = [registry.ext]; %and get their extensions

allfiles = dir(fullfile('D:\Uni\Sem2\DIP\DIP_Final\train3', '*.*')); 
isimage = arrayfun(@(f) ~f.isdir && any(strcmp(regexp(f.name, '\.(.*)$', 'tokens', 'once'), extensions')), allfiles);
imagefiles = allfiles(isimage);
i=1;
for imagefile = imagefiles'
    Im = imread(fullfile('D:\Uni\Sem2\DIP\DIP_Final\train3', imagefile.name));
    if size(Im,3)==3
        Im=rgb2gray(Im);
    end
    [n,m]=size(Im);
    for k=1:n
        for j=1:m
            if Im(k,j)<40
                Im(k,j)=0;
            end
        end
    end
    Im=imresize(Im,[512,512]);
    Im=reshape(Im,1,512*512);
    Im=double(Im);
    x_train(i,:)=Im;
    i=i+1;
end
x_train1=x_train;
x_train2=x_train(1:6,:);

[m,n]=size(x_train);
a={'Benign','Malignant','Normal','Tumor'};
i=1;
for k=1:3:(m+3)
    k=k-1;
        for j=1:3
         label{1,j}=a{i};
         y_train((k+j),1)=label(1,j);
        end
        i=i+1;
end

y_train1=[y_train(10:12,:);y_train(10:12,:);y_train(7:9,:)];
y_train2=y_train(1:6,:);

%% Applying Anistropic Diffusion Filter
a=[];
for i=1:9
    a=reshape(x_train(i,:),512,512);
    diff=imdiffusefilt(a,'NumberOfIterations',3);
    dif_img(i,:)=reshape(diff,1,512*512); 
end

%% Feature extracting by Stationary Wavelet Transform
a=[];
for i=1:9
    
    a=reshape(dif_img(i,:),512,512);
    [A,H,V,D] = swt2(a,5,'haar');
    A=wcodemat(A(:,:,5),512);
    H=wcodemat(H(:,:,5),512);
    V=wcodemat(V(:,:,5),512);    
    D=wcodemat(D(:,:,5),512);
    DWT_feat= [A,H,V,D];  
    % normalization
    DWT_feat=double(DWT_feat);
    DWT_feat=(DWT_feat-min(DWT_feat(:)))/(max(DWT_feat(:))-min(DWT_feat(:)));
    [m,n]=size(DWT_feat);
    feat1(i,:)=reshape(DWT_feat,1,m*n);
end
feat2=feat1(1:6,:);


%% SVM Training

svm1 = fitcsvm(feat1,y_train1,'KernelFunction', 'linear');
svm2 = fitcsvm(feat2,y_train2,'KernelFunction', 'linear');

%% Testing Data
registry = imformats;               %get all image type supported by matlab
extensions = [registry.ext];        %and get their extensions

allfiles = dir(fullfile('D:\Uni\Sem2\DIP\DIP_Final\test3', '*.*')); 
isimage = arrayfun(@(f) ~f.isdir && any(strcmp(regexp(f.name, '\.(.*)$', 'tokens', 'once'), extensions')), allfiles);
imagefiles = allfiles(isimage);
i=1;
for imagefile = imagefiles'
    Im = imread(fullfile('D:\Uni\Sem2\DIP\DIP_Final\test3', imagefile.name));
    if size(Im,3)==3
        Im=rgb2gray(Im);
    end
    [n,m]=size(Im);
    for k=1:n
        for j=1:m
            if Im(k,j)<40
                Im(k,j)=0;
            end
        end
    end
    Im=imresize(Im,[512,512]);
    Im=reshape(Im,1,512*512);
    Im=double(Im);
    x_test(i,:)=Im;
    i=i+1;
end

x_test1=x_test;
x_test2=x_test(1:2,:);

[m,n]=size(x_test);
a={'Benign','Malignant','Normal','Tumor'};
i=1;
label=[];
for k=1:(m+1)
    k=k-1;
    label{1,1}=a{i};
    y_test((k+1),1)=label(1,1);
    i=i+1;
end

y_test1=[y_test(4,:);y_test(4,:);y_test(3,:)];
y_test2=y_test(1:2,:);


%% Applying Anistropic Diffusion Filter

a=[];
for i=1:3
    a=reshape(x_test(i,:),512,512);
    diff=imdiffusefilt(a,'NumberOfIterations',3);
    dif_img_test(i,:)=reshape(diff,1,512*512); 
end


%% Feature extracting for Test Samples
a=[];
for i=1:3
    a=reshape(dif_img_test(i,:),512,512);
    [A,H,V,D] = swt2(a,5,'haar');
    A=wcodemat(A(:,:,5),512);
    H=wcodemat(H(:,:,5),512);
    V=wcodemat(V(:,:,5),512);    
    D=wcodemat(D(:,:,5),512);
    DWT_feat= [A,H,V,D];
    % normalization
    DWT_feat=double(DWT_feat);
    DWT_feat=(DWT_feat-min(DWT_feat(:)))/(max(DWT_feat(:))-min(DWT_feat(:)));
    [m,n]=size(DWT_feat);
    feat1_test(i,:)=reshape(DWT_feat,1,m*n);
end
feat2_test=feat1_test(1:2,:);

%% SVM Prediction

[Label_p1, score1] = predict(svm1,feat1_test);
[siz,~]=size(Label_p1);
a=[];
a={'Tumor'};
k=1;
L=[];
for i=1:(siz-1)
    L{1}=Label_p1{i,:};
    if isequal(L,a)
        N(i,:)=1;
        [L, score2] = predict(svm2,feat2_test(i,:));
        Label_p2{k,:}=L{1};
        k=k+1;
    else 
        N(i,:)=0;
        k=k+1;
    end
end
L=[];

%% Accuracy
a=[];
a1=[];
a2=[];
for i=1:3
    if isequal(Label_p1{i,:},y_test1{i,:})
        a1(i,1)=1;
    else
        a1(i,1)=0;
    end
end
[siz,~]=size(N);
k=1;
for i=1:siz
    if N(i,:)==1
        y_test2_new{k,:}=y_test2{i,:};
        k=k+1;
    else
        y_test2_new{k,:}=[];
        k=k+1;
    end
end

tf = isempty(y_test2_new);
if isequal(tf,1)
    return
else
[siz,~]=size(y_test2_new);
for i=1:siz
    if isequal(Label_p2{i,:},y_test2_new{i,:})
        a2(i,1)=1;
    else
        a2(i,1)=0;
    end
end
end

j=1;
for i=1:2
    if a1(i,1)==0
        I(j,1)=i;
        j=j+1;
    else 
        I(j,1)=0;
    end
end

M1=mean(a1);

tf = isempty(I);
if isequal(tf,1)
    return
else
    [g,~]=size(I);
    for i=1:2
        for j=1:g
            u=I(j,1);
            if i==u && u~=0
                a2(i,1)=0;
            end
        end
    end
end

M2=mean(a2);
Accuracy_TestSet=((M1+M2)/2)*100
        
%% Designing 

promptMessage = sprintf('Do you want to see a sample and its category?');
titleBarCaption = 'Continue';
buttonText = questdlg(promptMessage, titleBarCaption,'Yes','No','Yes');
if strcmpi(buttonText, 'No')
	return;
else
    Im=[];
    [filename, pathname]=uigetfile( {'*.jpg';'*.jpeg';'*.gif';'*.png';'*.bmp'},'Select file');
    if isequal(filename,0) || isequal(pathname,0)
        uiwait(msgbox ('User pressed cancel','failed','modal')  )
        return;
    else
        Im = imread(fullfile(pathname, filename));
        if size(Im,3)==3
            Im=rgb2gray(Im);
        end
        [n,m]=size(Im);
        for k=1:n
            for j=1:m
                if Im(k,j)<40
                    Im(k,j)=0;
                end
            end
        end
        Im=imresize(Im,[512,512]);
        Im=double(Im);
        %% otsu's thresholding and morphological operations
        strel1=strel('Line',2,0);
        strel2=strel('Line',2,90);
        strel3=strel('diamond',4);
        strel4=strel('diamond',3);
        strel5=strel('diamond',2);
        strel6=strel('diamond',1);
        T= graythresh(Im);
        img1=imbinarize(Im,T); 
        %% extracting tumor
        strel2=strel('Line',3,0);
        strel2=strel('Line',3,90);
        e=edge(img1,'sobel');
        dilate=imdilate(e,[strel1 strel2],'full');
        no_holes=imfill(dilate,'holes');
        erode=imerode(no_holes,strel3);
        erode=imerode(erode,strel4);
        erode=imerode(erode,strel4);
        erode=imerode(erode,strel5);
        erode=imerode(erode,strel5);
        erode=imerode(erode,strel5);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imresize(erode,[512,512]);
        without_skull=Im.*erode;
        T = 200;
        img1 = without_skull > T;
        t=bwareafilt(img1,1);
        boundaries{1} = bwboundaries(t,'noholes');
        figure;
        imshow(reshape(Im,512,512),[]);
        hold on
        p=boundaries{1};
        numberOfBoundaries = size(p, 1);
        for k = 1 : numberOfBoundaries
            thisBoundary = p{k};
            % Note: since array is row, column not x,y to get the x, we need to use the second column of 'thisBoundary'.
            plot(thisBoundary(:,2), thisBoundary(:,1), 'r', 'LineWidth', 2);
        end
        %% Applying Anistropic Diffusion Filter
        diff=imdiffusefilt(Im,'NumberOfIterations',3);
        dif_img=reshape(diff,1,512*512); 
        %% Feature extracting for Test Samples
        a=[];
        a=reshape(dif_img,512,512);
        [A,H,V,D] = swt2(a,5,'haar');
        A=wcodemat(A(:,:,5),512);
        H=wcodemat(H(:,:,5),512);
        V=wcodemat(V(:,:,5),512);
        D=wcodemat(D(:,:,5),512);
        DWT_feat= [A,H,V,D];
        % normalization
        DWT_feat=double(DWT_feat);
        DWT_feat=(DWT_feat-min(DWT_feat(:)))/(max(DWT_feat(:))-min(DWT_feat(:)));
        [m,n]=size(DWT_feat);
        DWT_feat=reshape(DWT_feat,1,m*n);
        [L_p1, ~] = predict(svm1,DWT_feat);
        a={'Tumor'};
        L=[];
        if isequal(L_p1,a)
            [L_p2,~] = predict(svm2,DWT_feat);
            a{1}=L_p2{1};
            h = msgbox(sprintf('It belongs to " %s " class',a{:}));
        else
            a{1}=L_p1{1};
            h = msgbox(sprintf('It belongs to " %s " class ',a{:}));
        end
    end
end


