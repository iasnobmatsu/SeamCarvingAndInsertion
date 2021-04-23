close all;    

% read in image
I = im2double(imread('img1.jpg'));
I_gray = rgb2gray(I);


% get mask of the region in image that should not be shrinked/expanded and
% should be protected 
mask = getMask(I);
imshow(mask);
%%


%%
% img 1 https://en.wikipedia.org/wiki/Johannes_Vermeer#/media/File:Johannes_Vermeer,_Girl_with_the_Red_Hat,_c._1665-1666,_NGA_60.jpg
% img 2 https://en.wikipedia.org/wiki/Water_Lilies_(Monet_series)#/media/File:WLA_lacma_Monet_Nympheas.jpg

%% detete vertical seams when protecting a region
a = delete_seam_vertical_mask(I, 100, true, mask);
imwrite(a, "img1_shrinked_v.jpg");

%% detete horizontal seams when protecting a region
a = delete_seam_horizontal_mask(I, 50, true, mask);
imwrite(a, "img1_shrinked_h.jpg");

%% inserting vertical seam when protecting a region
a = insert_seam_vertical_mask(I, 100, true, mask);
imwrite(a, "img1_inserted_v.jpg");

%% inserting horizontal seam when protecting a region
a = insert_seam_horizontal_mask(I,70, true, mask);
imwrite(a, "img1_inserted_h.jpg");

%%

%%


% given an image and a seam (matrix same size as image with 1 and 0. 1
% indicate position of the seam), draw a read line representing the seam on
% the image
function draw_seam(I, s)
            r = I(:,:,1);
            g = I(:,:,2);
            b = I(:,:,3); 
            r(s==1) = 1;
            g(s==1) = 0;
            b(s==1) = 0;
            I_n = cat(3, r, g, b);
            imshow(I_n);
end


% delete a specified number of vertical seams while protecting certain masked region
% the graphics argument is true/false specifying if should show the
% process of deleting seams.
function newI = delete_seam_vertical_mask(I, num, graphics, mask)

    hold on;
    for i = 1:num
        I_gray = rgb2gray(I);
        e = entropyfilt(I_gray); % use entropy as energy function
        e(mask==1)=50; % for pixels inside the protected mask, increase energy so this part will not be altered
        s = find_seam_vertical(e); % find the seam
        if (graphics) 
              draw_seam(I, s); 
        end
        I = delete_seam_vertical_once(I,s); % delete seam
        mask = delete_seam_vertical_once(mask, s); % also delete seam from the mask to make sizes match
        if (graphics)
            imshow(I);
        end

    end
    hold off;
    newI = I;
    imshow(newI);
end


% delete a specified number of horizontal seams while protecting certain masked region
% This function works similarly to the delete_seam_vertical_mask function so I did not
% comment inside this function
function newI = delete_seam_horizontal_mask(I, num, graphics, mask)
    hold on;
    for i = 1:num
        I_gray = rgb2gray(I);
        e = entropyfilt(I_gray);
        e(mask==1)=50;
        s = find_seam_horizontal(e);
        if (graphics)
              draw_seam(I, s);
        end
        I = delete_seam_horizontal_once(I,s);
        mask = delete_seam_horizontal_once(mask, s);
        if (graphics)
            imshow(I);
        end

    end
    hold off;
    newI = I;
    imshow(newI);
end


% delete a specified number of vertical seams without protecting certain masked region
% This function works similarly to the delete_seam_vertical_mask function so I did not
% comment inside this function
function newI = delete_seam_vertical(I, num, graphics)
    hold on;
    for i = 1:num
        I_gray = rgb2gray(I);
        e = entropyfilt(I_gray);
        s = find_seam_vertical(e);
        if (graphics)
              draw_seam(I, s);
        end
        I = delete_seam_vertical_once(I,s);
        if (graphics)
            imshow(I);
        end

    end
    hold off;
    newI = I;
    imshow(newI);
end


% Delete a specified number of horizontal seams without protecting certain masked region
% This function works similarly to the delete_seam_vertical_mask function so I did not
% comment inside this function
function newI = delete_seam_horizontal(I, num, graphics)
    hold on;
    for i = 1:num
        I_gray = rgb2gray(I);
        e = entropyfilt(I_gray);
        s = find_seam_horizontal(e);
        if (graphics)
              draw_seam(I, s);
        end
        I = delete_seam_horizontal_once(I,s);
        if (graphics)
            imshow(I);
        end

    end
    hold off;
    newI = I;
    imshow(newI);
end


% insert a specified number of vertical seams while protecting certain masked region
function newI = insert_seam_vertical_mask(I, num, graphics, mask)
    hold on;
    [rec, ind_rec] = seam_vertical_record_mask(I, num, mask); % record where the specified num of seams will be if they would be deleted
    ind_matrix = zeros(size(I(:,:,1))); % make an index matrix corresponding to pixel indes in original image
    for i = 1:size(I, 1)
       ind_matrix(i, :) = [1:1:size(I,2)];
    end
    for r = 1:num
        [h,w,t] = size(I);
        seam = zeros(size(ind_matrix)); % access each seam based on saved index
        
        
        % reverse engineer index in the new current image based on index in
        % saved record
        for i =1: size(ind_matrix,1)
            ind_mat_row=ind_matrix(i, :); % find index row index matrox
            saved_ind=ind_rec{r}(i); % find saved index
            actual_ind=find(ind_mat_row==saved_ind); % correct index is the index of value in the index matrix
            seam(i, actual_ind)=1;
        end
        
        if (graphics)
          draw_seam(I, seam); % draw seam
        end
        I = padarray(I, [0,1],0, "post"); % add 1 pixel column to image which will make place for the inserted seam
        ind_matrix = padarray(ind_matrix, [0,1], 0, "post"); % add to index matrix as well
        
        % insert seam pixels row by row, each row as one new pixel
        for i = 1:h
           [v, seam_ind] = max(seam(i,:)); % find the seam pixel and it's index
           
           % start from the last(rightmost) pixel which is closest to the empty new column,
           % move the pixel on the left of the current pixel to the
           % position of current pixel.
           % after this for loop, the seam positions would be empty, and
           % everything on one side of the seam would be moved 1 pixel
           % right
           for j = (w):-1:seam_ind
               topind = min (j+1, w+1); 
               I(i,topind,:) = I(i,j,:); 
               ind_matrix(i,topind,:) = ind_matrix(i,j,:); 
           end
           
           % bound seam index so it does not exceed the image range
           upper = min(seam_ind+2, w);
           lower = max(seam_ind-2, 1);
           
           % fill in the seam positions with the average of the 3 pixel
           % alongside the seam
           I(i, seam_ind, :)=(I(i, seam_ind,:)+I(i, upper,:)+I(i, lower,:))/3;
           ind_matrix(i, seam_ind, :)=0;
           
        end
        if (graphics)
             imshow(I);
        end
    end
    hold off;
    newI = I;
    imshow(newI);
end

% insert a specified number of horizontal seams while protecting certain masked region
% This function works similarly to the insert_seam_vertical_mask function so I did not
% comment inside this function
function newI = insert_seam_horizontal_mask(I, num, graphics, mask)
    hold on;
    [rec, ind_rec] = seam_horizontal_record_mask(I, num, mask);
    ind_matrix = zeros(size(I(:,:,1)));
    
    for i = 1:size(I, 2)
       ind_matrix(:,i) = [1:1:size(I,1)];
    end
    
    for r = 1:num
        [h,w,t] = size(I);
         seam = zeros(size(ind_matrix));
        
        for i =1: size(ind_matrix,2)
            ind_mat_row=ind_matrix(:,i);
            saved_ind=ind_rec{r}(i);
            actual_ind=find(ind_mat_row==saved_ind);
            seam(actual_ind,i)=1;
        end
        
        
        if (graphics)
          draw_seam(I, seam);
        end
        
        I = padarray(I, [1,0],1, "post");
        ind_matrix = padarray(ind_matrix, [1,0], 0, "post");
        
        
        for i = 1:w
           [v, seam_ind] = max(seam(:,i));
           for j = h:-1:seam_ind
               topind = min (j+1, h+1);
               I(topind,i,:) = I(j,i,:); 
               ind_matrix(topind,i,:) = ind_matrix(j,i,:); 
           end
           upper = min(seam_ind+2, h);
           lower = max(seam_ind-2, 1);
           I(seam_ind,i, :)=(I(seam_ind,i,:)+I(upper,i,:)+I(lower,i,:))/3;
           ind_matrix(seam_ind,i, :)=0;
        end
    
        
      
        if (graphics)
                imshow(I);
        end
    end
    hold off;
    newI = I;
    imshow(newI);
    
end


% insert a specified number of vertical seams without protecting certain masked region
% This function works similarly to the insert_seam_vertical_mask function so I did not
% comment inside this function
function newI = insert_seam_vertical(I, num, graphics)
    hold on;
    [rec, ind_rec] = seam_vertical_record(I, num);
    ind_matrix = zeros(size(I(:,:,1)));
    for i = 1:size(I, 1)
       ind_matrix(i, :) = [1:1:size(I,2)];
    end
    for r = 1:num
        [h,w,t] = size(I);
        seam = zeros(size(ind_matrix));
        
        for i =1: size(ind_matrix,1)
            ind_mat_row=ind_matrix(i, :);
            saved_ind=ind_rec{r}(i);
            actual_ind=find(ind_mat_row==saved_ind);
            seam(i, actual_ind)=1;
        end
        
        if (graphics)
          draw_seam(I, seam);
        end
        I = padarray(I, [0,1],0, "post");
        ind_matrix = padarray(ind_matrix, [0,1], 0, "post");
        
        for i = 1:h
          [v, seam_ind] = max(seam(i,:));
        
           for j = (w):-1:seam_ind
               topind = min (j+1, w+1);
               I(i,topind,:) = I(i,j,:); 
               ind_matrix(i,topind,:) = ind_matrix(i,j,:); 
           end
           upper = min(seam_ind+1, w);
           lower = max(seam_ind-1, 1);
           
           I(i, seam_ind, :)=(I(i, seam_ind,:)+I(i, upper,:)+I(i, lower,:))/3;
           ind_matrix(i, seam_ind, :)=0;
           
           
        end
        if (graphics)
             imshow(I);
        end
    end
    hold off;
    newI = I;
    imshow(newI);
end


% insert a specified number of horizontal seams without protecting certain masked region
% This function works similarly to the insert_seam_vertical_mask function so I did not
% comment inside this function
function newI = insert_seam_horizontal(I, num, graphics)
      hold on;
    [rec, ind_rec] = seam_horizontal_record(I, num);
    ind_matrix = zeros(size(I(:,:,1)));
    
    for i = 1:size(I, 2)
       ind_matrix(:,i) = [1:1:size(I,1)];
    end
    
    for r = 1:num
        [h,w,t] = size(I);
         seam = zeros(size(ind_matrix));
        
        for i =1: size(ind_matrix,2)
            ind_mat_row=ind_matrix(:,i);
            saved_ind=ind_rec{r}(i);
            actual_ind=find(ind_mat_row==saved_ind);
            seam(actual_ind,i)=1;
        end
        
        
        if (graphics)
          draw_seam(I, seam);
        end
        
        I = padarray(I, [1,0],1, "post");
        ind_matrix = padarray(ind_matrix, [1,0], 0, "post");
        
        
        for i = 1:w
           [v, seam_ind] = max(seam(:,i));
           for j = h:-1:seam_ind
               topind = min (j+1, h+1);
               I(topind,i,:) = I(j,i,:); 
               ind_matrix(topind,i,:) = ind_matrix(j,i,:); 
           end
           upper = min(seam_ind+2, h);
           lower = max(seam_ind-2, 1);
           I(seam_ind,i, :)=(I(seam_ind,i,:)+I(upper,i,:)+I(lower,i,:))/3;
           ind_matrix(seam_ind,i, :)=0;
        end
    
        
      
        if (graphics)
                imshow(I);
        end
    end
    hold off;
    newI = I;
    imshow(newI);
    
    
end

% find and record positions of a specified numver of vertical seams that would be
% deleted while protecting a region 
function [rec, ind_rec] = seam_vertical_record_mask(I, num, mask)
   rec = cell([num, 1]); % a cell for all the seams 
   ind_rec = cell([num, 1]);% a cell for indexes of all the seams mapping back to original image, will be an array
   ind_matrix = zeros(size(I(:,:,1))); % define an index matrix mapping to original image pixel index
   for i = 1:size(I, 1)
       ind_matrix(i,:) = [1:1:size(I,2)];
   end
   for i = 1:num
        I_gray = rgb2gray(I);
        e = entropyfilt(I_gray);% use entropy as energy function
        e(mask==1) = 50;% increase energy for pixels in protected region to reduce the chance they show up in a min energy path
        rec{i} = find_seam_vertical(e);% save each seam into a record
        ind_rec{i} = zeros(size(I,1),1); % save index (from original image) to a record, this is an array of all rows
        for j = 1: size(I,1)
            ind_mat_row = ind_matrix(j, :);
            ind_rec{i}(j)= ind_mat_row(rec{i}(j,:)==1);
        end
        ind_matrix = delete_seam_vertical_once(ind_matrix, rec{i}); % delete seam from index matrix as well
        I = delete_seam_vertical_once(I, rec{i}); % remove the current seam to get ready finding next seam to remove
        mask = delete_seam_vertical_once(mask, rec{i}); % remove seam on mask to so the mask will match size of image
   end

end


% find and record positions of a specified numver of horizontal seams that would be
% deleted while protecting a region
% This function works similarly to the seam_vertical_record_mask function so I did not
% comment inside this function
function [rec, ind_rec] = seam_horizontal_record_mask(I, num, mask)
   rec = cell([num, 1]); 
   ind_rec = cell([num, 1]);
   ind_matrix = zeros(size(I(:,:,1)));
   for i = 1:size(I, 2)
       ind_matrix(:,i) = [1:1:size(I,1)];
   end
   for i = 1:num
        I_gray = rgb2gray(I);
        e = entropyfilt(I_gray); 
        e(mask==1) = 50; 
        rec{i} = find_seam_horizontal(e);
        ind_rec{i} = zeros(size(I,2),1);
        for j = 1: size(I,2)
            ind_mat_row = ind_matrix(:,j);
            ind_rec{i}(j)= ind_mat_row(rec{i}(:, j)==1);
        end
        ind_matrix = delete_seam_horizontal_once(ind_matrix, rec{i});
        rec{i} = find_seam_horizontal(e); 
        I = delete_seam_horizontal_once(I, rec{i}); 
        mask = delete_seam_horizontal_once(mask, rec{i}); 
   end

end



% find and record positions of a specified numver of vertical seams that would be
% deleted without protecting a region
% This function works similarly to the seam_vertical_record_mask function so I did not
% comment inside this function
function [rec, ind_rec] = seam_vertical_record(I, num)
   rec = cell([num, 1]);
   ind_rec = cell([num, 1]);
   ind_matrix = zeros(size(I(:,:,1)));
   for i = 1:size(I, 1)
       ind_matrix(i, :) = [1:1:size(I,2)];
   end
   for i = 1:num
        I_gray = rgb2gray(I);
        e = entropyfilt(I_gray);
        rec{i} = find_seam_vertical(e);
        ind_rec{i} = zeros(size(I,1),1);
        for j = 1: size(I,1)
            ind_mat_row = ind_matrix(j, :);
            ind_rec{i}(j)= ind_mat_row(rec{i}(j,:)==1);
        end
        I = delete_seam_vertical_once(I, rec{i});
        ind_matrix = delete_seam_vertical_once(ind_matrix, rec{i});
   end

end




% find and record positions of a specified numver of horizontal seams that would be
% deleted without protecting a region
% This function works similarly to the seam_vertical_record_mask function so I did not
% comment inside this function
function [rec, ind_rec] = seam_horizontal_record(I, num)
   rec = cell([num, 1]); 
   ind_rec = cell([num, 1]);
   ind_matrix = zeros(size(I(:,:,1)));
   for i = 1:size(I, 2)
       ind_matrix(:,i) = [1:1:size(I,1)];
   end
   for i = 1:num
        I_gray = rgb2gray(I);
        e = entropyfilt(I_gray); 
        rec{i} = find_seam_horizontal(e);
        ind_rec{i} = zeros(size(I,2),1);
        for j = 1: size(I,2)
            ind_mat_row = ind_matrix(:,j);
            ind_rec{i}(j)= ind_mat_row(rec{i}(:, j)==1);
        end
        ind_matrix = delete_seam_horizontal_once(ind_matrix, rec{i});
        rec{i} = find_seam_horizontal(e); 
        I = delete_seam_horizontal_once(I, rec{i}); 
   end

end



% delete a vertical seam from a given image
function newI = delete_seam_vertical_once(I, seam)
    [h,w] = size(seam);
    for i = 1:h
       [v, seam_ind] = max(seam(i,:)); %  find the seam index/position
       % move every pixel after the seam one pixel forward
       for j = seam_ind:(w-1)
           I(i,j,:) = I(i,j+1,:); 
       end
    end
    % crop the last column from image since now it is empty
    newI = imcrop(I, [1, 1,w-2, h-1]);

end

% delete a horizontal seam from a given image. This function is similar to 
% function delete_seam_vertical_once so I did not comment inside
function newI = delete_seam_horizontal_once(I, seam)
    [h,w] = size(seam);
    for i = 1:w
       [v, seam_ind] = max(seam(:,i));
       for j = seam_ind:(h-1)
           I(j,i,:) = I(j+1,i,:); 
       end
    end
    newI = imcrop(I, [1, 1,w-1, h-2]);

end



% find a vertical seam given the energy same size as the image 
% seam returned is a matrix same size as image with 0/1 indicating a seam
% path
function seam = find_seam_vertical(e)
    [h,w] = size(e);
    path_energy = zeros(size(e)); % container for energy for every path
    path_energy(1,:) = e(1,:); % first row stay the same
    path = zeros(size(e));
    
    for i = 2:h % loop over rows
        for j = 1:w % loop over cols
            lower = max(1, j-1); % find lower bound for position of next pixel
            upper = min(w, j+1); % find upper bound for position of next pixel
           
            path_energy(i, j) = e(i, j)+ min(path_energy(i-1, lower:upper));  % get min energy path for the current pixel - add the energy of current pixel to pixel within bound with min energy on the prev step
            [min_e, index_e] = min(path_energy(i-1, lower:upper));  % find which direction is taken to achive current min energy
            index_e = index_e - 2;% adjust direction so if the new step stayed on the same row/col then index is 0, if it is 1 left/upper than -1, if it is 1 right/lower that 1.
            path(i, j) = index_e; % add current direction to path
        
        end
    end
    
  
    seam = zeros(size(e));  % actual matrix container for the seam. same size as image, all 0 except 1 at seam positions
    [min_fe, ind_fe] = min(path_energy(h,:)); % find min energy on the last row and fine the corresponding index
    seam(h, ind_fe) = 1;
    for i = h-1:-1:1
         % reverse-engineer index of each previous seam position
        prev_step_ind = path(i+1, ind_fe) + ind_fe;
        ind_fe = prev_step_ind;
        ind_fe = max(ind_fe, 1);
        ind_fe = min(ind_fe, w);
        seam(i, ind_fe) = 1;
    end 
end



% find a horizontal seam given the energy same size as the image 
% seam returned is a matrix same size as image with 0/1 indicating a seam
% path
function seam = find_seam_horizontal(e)
    [h,w] = size(e);
    path_energy = zeros(size(e)); % container for total energy of every path
    path_energy(:,1) = e(:,1); % first column of the energy matrix stay the same
    path = zeros(size(e));% container for direction to next step for every path
    
    for i = 2:w % loop over cols
        for j = 1:h % loop over rows
         
            % restrict so current pixel on each path is connected to at least
            % a corner of the last pixel
            lower = max(1, j-1); % find lower bound for position of next pixel
            upper = min(h, j+1); % find upper bound for position of next pixel
          
            path_energy(j,i) = e(j,i)+ min(path_energy(lower:upper,i-1));  % get min energy path for the current pixel - add the energy of current pixel to pixel within bound with min energy on the prev step
            [min_e, index_e] = min(path_energy(lower:upper,i-1)); % find which direction is taken to achive current min energy
            index_e = index_e - 2; % adjust direction so if the new step stayed on the same row/col then index is 0, if it is 1 left/upper than -1, if it is 1 right/lower that 1.
            path(j,i) = index_e; % add the step direction into the path container
        
        end
    end
    
  
    seam = zeros(size(e)); % actual matrix container for the seam. same size as image, all 0 except 1 at seam positions
    [min_fe, ind_fe] = min(path_energy(:, w)); % find min energy index at the end of path
    seam(ind_fe,w) = 1; 
    for i = w-1:-1:1
        % reverse-engineer index of each previous seam position
        prev_step_ind = path(ind_fe,i+1) + ind_fe; 
        ind_fe = prev_step_ind;
        ind_fe = max(ind_fe, 1);
        ind_fe = min(ind_fe, h);
        seam(ind_fe, i) = 1;
    end 
end









    

%

