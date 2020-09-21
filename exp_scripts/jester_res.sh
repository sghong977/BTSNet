# this script is for testing kinetics+Moment datasets

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

pre_path=' '  #'--pretrain_path results/mit_sknet26_M2__save_6.pth --n_pretrain_classes 339'
resume_path=' '
root_path='./'
annotation_path='./'
video_path='../../../NAS/DATA2/jester'

model=sknet   #resnet  sknet... 
depth=(50)  # 50 101
M=2

batch_size=128

dataset=resnet
n_classes=27
sample_duration=30
checkpoint=5
n_epochs=200
#multistep_milestones='[7,11,15]'  # ㅇㅣ거 opt 고쳤으니까 딴거 돌릴 떄 주의
learning_rate=0.1

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $resume_path $pre_path --learning_rate $learning_rate --n_epochs $n_epochs --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/sknet_$M$dataset$model${depth[$i]}$i.txt
done
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt
