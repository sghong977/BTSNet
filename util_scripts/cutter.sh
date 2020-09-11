# NO IT DOESNT WORK
#!/bin/bash
#1:File address,2:frm_rate,3:save_path

EXPECTED_ARGS=3
E_BADARGS=65

#if [ $# -lt $EXPECTED_ARGS ]
#then
#  echo "Usage: `basename $0` video frames/sec [size=256] save path"
#  exit $E_BADARGS
#fi

File_addr='yt-ZZQPszKg6Uc_1.mp4'   #1

NAME=${File_addr%.*}
FRAMES=30
BNAME='yt-ZZQPszKg6Uc_1/yt-ZZQPszKg6Uc_1'
#echo $BNAME
#mkdir -m 755 $BNAME
#ffmpeg -y -i $1 -r $FRAMES $3_%4d.jpg

ffmpeg -i $File_addr -qscale:v 2 -r $FRAMES  $BNAME_%4d.jpg