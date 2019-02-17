# rename all files in a directory
  
import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
    poke = "Ponyta"
      
    for filename in os.listdir(poke): 
        src = poke + "/" + filename
        dst = poke + "_" + str(i) + ".png"
        dst = poke + "/" + dst
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 

