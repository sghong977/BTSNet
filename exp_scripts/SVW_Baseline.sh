# this script is for testing kinetics+Moment datasets

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

GPU=6

pre_path=' '  #'--pretrain_path results/mit_sknet26_M2__save_6.pth --n_pretrain_classes 339'
resume_path=' '
root_path='./'
annotation_path='./csv_and_json/'
video_path='../../../raid/SVW'

model=slowfast   #resnet  sknet... 
depth=(50 101 152 200)  #101 
M=4              #
fuse_layer=TC    #
ops_type=O2      #

batch_size=32

dataset=SVW
n_classes=30
sample_duration=30

checkpoint=251
learning_rate=0.01        #
n_epochs=200               #
scheduler=SGDR         #

#plateau_patience=5
cardinality='--resnext_cardinality 32'

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU python main.py $cardinality --lr_scheduler $scheduler --learning_rate $learning_rate $resume_path $pre_path --n_epochs $n_epochs --ops_type $ops_type --fuse_layer $fuse_layer --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/logs_$scheduler$learning_rate$M$dataset$model${depth[$i]}$fuse_layer$ops_type$i.txt
done


# ResNet
model=resnet   #resnet  sknet... 
depth=(50 101)
learning_rate=0.1        #
scheduler=multistep         #
n_epochs=200

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU python main.py $cardinality --lr_scheduler $scheduler --learning_rate $learning_rate $resume_path $pre_path --n_epochs $n_epochs --ops_type $ops_type --fuse_layer $fuse_layer --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/logs_$scheduler$learning_rate$M$dataset$model${depth[$i]}$fuse_layer$ops_type$i.txt
done



model=resnext   #resnet  sknet... 
depth=(50 101)

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU python main.py $cardinality --lr_scheduler $scheduler --learning_rate $learning_rate $resume_path $pre_path --n_epochs $n_epochs --ops_type $ops_type --fuse_layer $fuse_layer --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/logs_$scheduler$learning_rate$M$dataset$model${depth[$i]}$fuse_layer$ops_type$i.txt
done
