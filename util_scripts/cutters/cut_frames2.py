#video to image
# Moment in time, validation
import cv2 
import os 


# to check,
# folder : ls -l | grep ^d | wc -l
# files : ls -l | grep ^- | wc -l

# Creating...../../../../nas/Public/Kinetics_700/cut/validation/alligator wrestling/rYtYJQF0xTU_000007_000017/image_00079.jpg
in_path = "../../../../NAS/DATA2/Kinetics_700/train/"
classes = os.listdir(in_path)
base_out_path='../../../../NAS/DATA2/Kinetics_700/cut/train/'

print(classes)

for c in classes:
    #list in this folder
    file_list = os.listdir(in_path + c)

    # creating a folder named data 
    if not os.path.exists(base_out_path+c): 
        os.makedirs(base_out_path+c)
    #--------- read one video ------------
    for f in file_list:
        if not os.path.exists(base_out_path + c + '/' + f[:-4]): 
            os.makedirs(base_out_path + c + '/' + f[:-4]) 
        print(c+'/'+f)
        cam = cv2.VideoCapture(in_path + '/' + c + '/' + f) 
        # frame 
        out_path = base_out_path + c + '/' + f[:-4]

        currentframe = 0
        while(True): 
            # reading from frame 
            ret,frame = cam.read() 
            if ret: 
                # if video is still left continue creating images 
                name = out_path + '/image_' + str(currentframe).zfill(5) + '.jpg'
                # writing the extracted images 
                cv2.imwrite(name, frame) 
                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1
            else: 
                break
        # Release all space and windows once done 
        cam.release() 
        cv2.destroyAllWindows() 