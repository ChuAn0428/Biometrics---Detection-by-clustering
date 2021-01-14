%--------------------------------------
% CSCI 59000 Biometrics - Detection by clustering
% Author: Chu-An Tsai
%--------------------------------------

clear;
PICs = ["Lihua.jpg" "Haiying.jpg"];

for numP = 1 : length(PICs)  
    fprintf('%d. For image ''%s''\n',numP,PICs(numP))
    prompt1 = ' (1) Enter the value for K: ';
    K = input(prompt1);
    image = imread(PICs(numP));
    
    % convert to 2D
    X = double(reshape(image,size(image,1)*size(image,2),3)); 
    N = length(X);
    numOfExec = 1;
    prompt2 = ' (2) Enter the number for iterations: ';
    numEnd = input(prompt2);
    
    clear totalXout;
    clear checkCentroids;
    clear totalCentroids;
    
    totalXout = zeros(N,1);
    protempXout = zeros(N,K);
    for numExe = numOfExec : numEnd
        randCentroid = sort(X(randperm(N,K),:));
        [Xout,Centroids] = kmeans(X,K,'Maxiter',300,'Start',randCentroid);
        
        %-----------------------------------
        %{   
        % graph to check class distribution (for K = 3)
        figure;
        params = { 'bs' , 'r^' , 'go' , 'md' , 'c+' };
        for i = 1:3 
            clust = find(Xout==i);
            plot3(X(clust,1),X(clust,2),X(clust,3),params{i});
            hold on 
        end 
        plot3(Centroids(:,1),Centroids(:,2),Centroids(:,3), 'kx', 'LineWidth',5);
        legend('Cluster 1','Cluster 2','Cluster 3','Centroids')
        title 'Cluster Assignments and Centroids'
        hold off 
        view(-137,10);
        grid on
        %}   
        %-----------------------------------
        
        % 2D with class label
        totalXout = [totalXout Xout]; 
        
        % record initial centroids for each iteration for checking 
        checkCentroids(:,:,numOfExec) = randCentroid; 
        
        % record final centroids for each iteration for checking
        totalCentroids(:,:,numOfExec) = Centroids;
        
        % 3D with class label and numofexec
        %M(:,:,numOfExec) = imageOut; 
    
        numOfExec = numOfExec + 1; 
    end
    
    % delete the first column (all zeros)
    totalXout(:,1) = [];
   
    % calculate the frequency
    for i = 1 : N
        for j = 1 : numEnd
            for k = 1 : K
                if totalXout(i,j) == k
                    protempXout(i,k) = protempXout(i,k) + 1;
                end
            end
        end    
    end
    
    % convert to probability
    proXout = protempXout / numEnd; 
    
    % convert to 3D (3 clusters)
    proC3Xout = reshape(proXout,size(image,1),size(image,2),K);
    
    % plot probability maps
    for i = 1 : K
        figure(numP);
        subplot(1,K,i)
        imagesc(proC3Xout(:,:,i))
        title(['The probability map for cluster #',num2str(i)])
    end
    
    %------------------- GR --------------------------
    
    % display the image
    figure(numP+2);
    
    % let user input face 
    axis ij;
    axis manual;
    [m,n] = size(PICs(numP)); 
    axis([0 m 0 n])
    imshow(PICs(numP));
    title 'Click in the face regions to input sample'
    
    % get the input coordinates
    coordinates_input = ginput(1);
    row = round(coordinates_input(2));
    column = round(coordinates_input(1));
    hold on
    
    % plot the point that user has clicked
    scatter(column,row, 'filled' , 'r' )
    hold off
    
    % get the R, G, B value for the input sample
    faceSample = double([image(row,column,1),image(row,column,2),image(row,column,3)]);
    
    % calculate the distance between the input sample and each cluster centroid
    d = zeros(K,1);
    for i = 1 : K
        d(i) = pdist([faceSample;Centroids(i,:)],'euclidean');
    end
    
    % get the index for the shortest distance to obtain the cluster number
    [M,Id] = min(d);
    proXoutSkin = proC3Xout(:,:,Id);
    
    %{
    % assign 1 or 0 according to the probability
    for i = 1 : size(image,1)
        for j = 1 : size(image,2)
            if proC3Xout(i,j,Id) > 0.5
                proXoutSkin(i,j) = 1;
            else 
                proXoutSkin(i,j) = 0;
            end
        end
    end
    %}
    
    % plot the skin region detected
    figure(numP+4);
    imshow(proXoutSkin);
    title 'All skin regions in the image'    
end