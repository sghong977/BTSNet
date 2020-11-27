#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

GPU=4


# NO ROOT AND VIDEO PATH
annotation_path='/raid/video_data/epic/slowfast_frames/annotation/epic_noun.json'   #epic_fullnoun.json'

model=resnext   #resnet  sknet... 
depth=(101 50)

M=4              #
fuse_layer=TC    #
ops_type=O2      #


batch_size=32

dataset=epic
n_classes=352   # noun
sample_duration=30

checkpoint=250             #
learning_rate=0.01        #
n_epochs=250               #
scheduler=multistep         #

#plateau_patience=5
cardinality='--resnext_cardinality 32'

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU python main.py $cardinality --learning_rate $learning_rate $resume_path $pre_path --n_epochs $n_epochs --ops_type $ops_type --fuse_layer $fuse_layer --M $M --annotation_path $annotation_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/logs_$scheduler$learning_rate$M$dataset$model${depth[$i]}$fuse_layer$ops_type$i.txt
#    CUDA_VISIBLE_DEVICES=$GPU python main.py --plateau_patience $plateau_patience $cardinality --lr_scheduler $scheduler --learning_rate $learning_rate $resume_path $pre_path --n_epochs $n_epochs --ops_type $ops_type --fuse_layer $fuse_layer --M $M --annotation_path $annotation_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/logs_$scheduler$learning_rate$M$dataset$model${depth[$i]}$fuse_layer$ops_type$i.txt
done

