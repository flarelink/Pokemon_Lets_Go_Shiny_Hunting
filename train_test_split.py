import glob, os


dataset_path = '/home/flarelink/Documents/Github_Projects/Pokemon_Lets_Go_Shiny_Hunting/images/Shiny_Pokemon/Gastly'

labels_path = '/home/flarelink/Documents/Github_Projects/Pokemon_Lets_Go_Shiny_Hunting/labels'

# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')  
file_test = open('test.txt', 'w')

# Populate train.txt and test.txt
counter = 1  
index_test = round(100 / percentage_test)  

# check files in labels path
for pathAndFilename_labels in glob.iglob(os.path.join(labels_path, "*.txt")):  
    title_labels, ext_labels = os.path.splitext(os.path.basename(pathAndFilename_labels))

    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.png")):  
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        # check if labels txt == image png
        if(title_labels == title):

            if counter == index_test+1:
                counter = 1
                file_test.write(dataset_path + "/" + title + '.png' + "\n")
            else:
                file_train.write(dataset_path + "/" + title + '.png' + "\n")
                counter = counter + 1
