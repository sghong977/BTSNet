#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

GPU=5

root_path='./'
annotation_path='./csv_and_json/ucf101_01.json'
video_path='../../../raid/video_data/ucf101/ucf101_videos'

model=btsnet   #resnet  sknet... 
depth=(50 101)
M=4              #
fuse_layer=TC    #
ops_type=O2      #


batch_size=32

dataset=ucf101
n_classes=101
sample_duration=30

checkpoint=200
learning_rate=0.1        #
n_epochs=200
scheduler=multistep         #

#plateau_patience=5
cardinality='--resnext_cardinality 32'
card=8

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU python main.py $cardinality --lr_scheduler $scheduler --learning_rate $learning_rate $resume_path $pre_path --n_epochs $n_epochs --ops_type $ops_type --fuse_layer $fuse_layer --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint 
done
