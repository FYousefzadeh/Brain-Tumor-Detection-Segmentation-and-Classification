clc;
clear;
close all;
%% Main Code
promptMessage = sprintf('Do you want to see a sample and its category?');
titleBarCaption = 'Continue';
buttonText = questdlg(promptMessage, titleBarCaption,'Yes','No','Yes');
if strcmpi(buttonText, 'No')
	return;
else
    Im=[];
    [filename, pathname]=uigetfile( {'*.png';'*.jpg';'*.jpeg';'*.gif';'*.bmp'},'Select file');
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
        Im=imresize(Im,[200,200]);
        Im=double(Im);
        %% otsu's thresholding and morphological operations
        T= graythresh(Im);
        img1=imbinarize(Im,T); 
        %% extracting tumor
        strel1=strel('Line',2,0);
        strel2=strel('Line',2,90);
        strel3=strel('diamond',4);
        strel4=strel('diamond',3);
        strel5=strel('diamond',2);
        strel6=strel('diamond',1);
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
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imerode(erode,strel6);
        erode=imresize(erode,[200,200]);
        without_skull=Im.*erode;
        T = 180;
        img1 = without_skull > T;
        t=bwareafilt(img1,1);
        boundaries{1} = bwboundaries(t,'noholes');
        figure;
        imshow(reshape(Im,200,200),[]);
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
        dif_img=reshape(diff,1,200*200); 
        %% Feature extracting for Test Samples
        a=[];
        a=reshape(dif_img,200,200);
        [A,H,V,D] = swt2(a,3,'haar');
        A=wcodemat(A(:,:,3),200);
        H=wcodemat(H(:,:,3),200);
        V=wcodemat(V(:,:,3),200);
        D=wcodemat(D(:,:,3),200);
        DWT_feat= [A,H,V,D];
        % normalization
        DWT_feat=double(DWT_feat);
        DWT_feat=(DWT_feat-min(DWT_feat(:)))/(max(DWT_feat(:))-min(DWT_feat(:)));
        [m,n]=size(DWT_feat);
        DWT_feat=reshape(DWT_feat,1,m*n);
        %% Prediction
        L_p1 = classifySVM1_mex(DWT_feat);
        a={'Tumor'};
        L=[];
        if isequal(L_p1,a)
            L_p2 = classifySVM2_mex(DWT_feat);
            a{1}=L_p2{1};
            h = msgbox(sprintf('It belongs to " %s " class',a{:}));
        else
            a{1}=L_p1{1};
            h = msgbox(sprintf('It belongs to " %s " class ',a{:}));
        end
    end
end