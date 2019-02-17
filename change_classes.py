import glob, os
import shutil


labels_path = '/home/flarelink/Documents/Github_Projects/Pokemon_Lets_Go_Shiny_Hunting/labels'


# check files in labels path
for pathAndFilename_labels in glob.iglob(os.path.join(labels_path, "*.txt")):  
    title_labels, ext_labels = os.path.splitext(os.path.basename(pathAndFilename_labels))

    # read file    
    from_file = open(pathAndFilename_labels)
    
    # grab first line
    line = from_file.readline()
    line = line.split(' ')
    
    # find which pokemon it is
    poke = title_labels.split('_')

    if(poke[0] == 'Gastly'):
        line[0] = '2'
        line = ' '.join(line)
        to_file = open(pathAndFilename_labels, mode='w')
        to_file.write(line)
        shutil.copyfileobj(from_file, to_file)
        
    elif(poke[0] == 'Ponyta'):
        line[0] = '1'
        line = ' '.join(line)
        to_file = open(pathAndFilename_labels, mode='w')
        to_file.write(line)
        shutil.copyfileobj(from_file, to_file)



