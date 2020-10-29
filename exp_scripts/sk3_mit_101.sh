# this script is for testing kinetics+Moment datasets

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

pre_path=' '  #'--pretrain_path results/mit_sknet26_M2__save_6.pth --n_pretrain_classes 339'
resume_path=' '  #' --resume_path results/mit_sknet26_M2__save_6.pth '
root_path='./'
annotation_path='./'
video_path='../../../raid/Moment/Moments_in_Time_256x256_30fps'

model=sknet3   #resnet  sknet... 
depth=(50)  # 50 101
M=4

batch_size=128

dataset=mit
n_classes=339
sample_duration=30
checkpoint=1
n_epochs=200

scheduler='--lr_scheduler plateau'

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6 python main.py $scheduler --n_epochs $n_epochs --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/sknet_$M$dataset$model${depth[$i]}$i.txt
done
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt
