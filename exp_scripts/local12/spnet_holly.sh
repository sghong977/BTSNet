# this script is for testing kinetics+Moment datasets

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

pre_path=' '  #'--pretrain_path results/mit_sknet26_M2__save_6.pth --n_pretrain_classes 339'
resume_path=' '
root_path='./'
annotation_path='./csv_and_json/'
video_path='../../../data/video_dataset/Hollywood2'

model=resnet   #spnet  sknet... 
depth=(101)   # 26
M=4              #
fuse_layer=TC    #
ops_type=O2      #

batch_size=32

dataset=hollywood2
n_classes=12
sample_duration=30
checkpoint=100
epoch=200

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python main.py --n_epochs $epoch --ops_type $ops_type --fuse_layer $fuse_layer --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/logs_$M$dataset$model${depth[$i]}$fuse_layer$ops_type$i.txt
done
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt