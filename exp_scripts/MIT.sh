# this script is for testing kinetics+Moment datasets

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

root_path='./'
annotation_path='../../../raid/Moment/Moments_in_Time_256x256_30fps/trainingSet.csv'
video_path='../../../raid/Moment/Moments_in_Time_256x256_30fps'

model=sknet   #resnet  sknet... 
depth=(26)  # 50 101
M=2

batch_size=32

dataset=mit
n_classes=339
sample_duration=30
checkpoint=200


i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint #> logs/sknet_$M$model${depth[$i]}$i.txt
done
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt