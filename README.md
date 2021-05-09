# SeamCarvingAndInsertion
final project for computational photography

## Aim
In this project, I implemented methods to do vertical carving, horizontal carving, vertical insertion, and horizontal insertion in MATLAB. Before deleting or inserting seams, a mask can be manually defined on the input image that specifies an area in the image that is prone to artifacts and thus should be protected from changes.

## Seam Deletion 
Horizontal Carving (removing vertical seams): The function find_seam_vertical uses dynamic programing to retrieve a optimum seam that starts at the top of an image and ends at the bottom. The function keeps two records, one is for the minimum energy at each pixel, the other for the direction of each step. The energy function used in this project is entropy. Using these two records, a matrix same size as the image with 0 indicate not a pixel from the seam and 1 indicating a pixel from the seam can be constructed. The function delete_seam_vertical_mask loops over the image a specified amount of times to remove multiple seams. Vertical Carving (removing horizontal seams) is achieved similarly. 

## Seam Insertion
Horizontal Insertion (adding vertical seams): The function seam_vertical_record_mask uses the methods to find seams from seam deletion to keep a record of multiple seams that would be removed if a seam deletion is performed. The record is of the index of pixels in the original image that the seams are removing (so if a 5th seam is found, it would not keep track of the new index of the seam from the shrinked image after removing the previous 4 seams, but rather keeping track of positions in the original image). Using the seam indexes, the function seam_vertical_record_mask will duplicate seams that would be deleted in the performed order. Because inserting seams also changes the index of pixels in the original image, a matrix of original pixel index is stored, and the seam inserted is always on the index of the original image pixels. Thus, in the animation of the seam insertion, there will be times that the red line indicating seams became disconnected. This is because previous seams inserted have changed pixel positions in the image, and in order to insert corresponding seams, their pixel positions need to be adjusted too. The duplicated seam is the average of the three pixels surrounding it in the horizontal direction. Vertical Insertion (adding horizontal seams) is achieved similarly. 

## Some Challenges
While implementing seam deletions, the most challenging aspect is to correctly set up dynamic program to find the seam. Because seam pixels need to be connected to previous pixels (can only be off by 1 pixel), when dynamically going through all pixels in an image, the range that the next step pixel can take is restricted to between +1 and -1 of the current pixel position. It also took some tries to figure out that using entropy as the energy function is relatively stable and easy to implement. Other functions that detect edges that I tried did not work as well in seam carving. 
In my opinion, seam insertion is more challenging than seam carving. The intuitive solution is to find a seam, duplicate it, then find a new seam in the stretched image, and again duplicated it. This repeated seam insertion process, however, would not work because it would just be the same seam being duplicated again and again. Therefore, there is the need find at once set of all seams that would be removed if performing a seam deletion, then reverse-engineering these seams and duplicate each of them. It is also challenging to figure out how to correctly duplicate seams at correct pixels. The seams, if finding them the seam way as in seam deletion, would not work, because the image sizes have changed. Therefore, it is necessary to keep track of seams in the pixel index or positions in the original image’s perspective. And when inserting the duplicated seams, it is necessary again to use the original image’s indexes. 

The other challenging part of the project is to find out a way to minimize artifacts, especially on images where there is a specific object to keep intact. The solution I came up with is to use the getMask function that we have used in multiple other assignments. Using a user-specified black and white mask of the image, it is possible to increase energy on the masked regions when finding seams. Thus, the seam-finding algorithm will seek to avoid these regions.

## Seam Carving and Insertion Processes

![gif](carving/seam-carving-demonstration-video.gif)

## Examples

original image

![](carving/set4_original.jpg)

removing 90 pixels horizontally

![](carving/set4_carved3.jpg)

inseting 30 pixels horizontally

![](carving/set4_carved7.jpg)

[original image](https://jojo.fandom.com/wiki/Hirohiko_Araki_JoJo_Exhibition_2012?file=Exhib8.jpg)

![](carving/set1_original.jpg)

removing 90 pixels vertically 

![](carving/set1_carved6.jpg)

inseting 30 pixels vertically

![](carving/set1_carved8.jpg)


