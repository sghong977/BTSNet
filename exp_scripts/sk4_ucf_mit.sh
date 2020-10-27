#TEST 0

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

pre_path=' --pretrain_path results/mit_sknet350_M4__save_101.pth --n_pretrain_classes 339 '
root_path='./'
annotation_path='./ucf101_01.json'
video_path='../../../raid/video_data/ucf101/ucf101_videos'

model=sknet3   #resnet  sknet... 
depth=(50)
M=4

batch_size=32

dataset=ucf101
n_classes=101
sample_duration=30
checkpoint=100
epoch=200

# finetuning 시작하는 곳임. 이거 아예 없어야만 전부 학습됨
fc_begin=' ' #--ft_begin_module fc '

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=7 python main.py $scheduler $pre_path --n_epochs $epoch --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes $fc_begin --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/pret_sknet3_$M$dataset$model${depth[$i]}$i.txt
done
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt




#--------------------
pre_path=' --pretrain_path results/mit_sknet350_M4__save_101.pth --n_pretrain_classes 339 '
resume_path=' '
root_path='./'
annotation_path='./'
video_path='../../../raid/Hollywood2'

model=sknet3   #resnet  sknet... 
depth=(50) 
M=4

batch_size=32

dataset=hollywood2
n_classes=12
sample_duration=30
checkpoint=100
epoch=200

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=7 python main.py $scheduler $pre_path --n_epochs $epoch --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes $fc_begin --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/pret_sknet3_$M$dataset$model${depth[$i]}$i.txt
done