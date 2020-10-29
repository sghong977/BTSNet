# scripts for New version!

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

pre_path=' '  #'--pretrain_path results/mit_sknet26_M2__save_6.pth --n_pretrain_classes 339'
resume_path=' --resume_path results/charades_sknet350_M4__save_100.pth '
root_path='./'
annotation_path='./'
video_path='../../../raid/Charades'

model=spnet   #resnet  sknet... 
depth=(50) 
M=4              #
fuse_layer=TC    #
ops_type=O2      #

batch_size=2

dataset=charades
n_classes=157
sample_duration=30
checkpoint=20
n_epochs=200   # no lr drop. 
learning_rate=0.01
scheduler='--lr_scheduler plateau'

#multistep_milestones='[7,11,15]'  # ㅇㅣ거 opt 고쳤으니까 딴거 돌릴 떄 주의

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=7 python main.py $scheduler --learning_rate $learning_rate $resume_path $pre_path --n_epochs $n_epochs --ops_type $ops_type --fuse_layer $fuse_layer --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint #> logs/logs_$M$dataset$model${depth[$i]}$fuse_layer$ops_type$i.txt
done