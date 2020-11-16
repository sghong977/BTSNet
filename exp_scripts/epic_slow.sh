#TEST 0

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

annotation_path='/raid/video_data/epic/slowfast_frames/annotation/epic_noun.json'   #epic_fullnoun.json'

model=resnet   #resnet  sknet... 
depth=(101 50)

M=4              #
fuse_layer=TC    #
ops_type=O2      #


batch_size=32

dataset=epic
n_classes=352   # noun
sample_duration=30

checkpoint=200
learning_rate=0.01
n_epochs=200


i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python main.py $scheduler --learning_rate $learning_rate $resume_path $pre_path --n_epochs $n_epochs --ops_type $ops_type --fuse_layer $fuse_layer --M $M --annotation_path $annotation_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/logs_$M$dataset$model${depth[$i]}$fuse_layer$ops_type$i.txt
done
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt




model=slowfast   #resnet  sknet... 
depth=(50)

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python main.py $scheduler --learning_rate $learning_rate $resume_path $pre_path --n_epochs $n_epochs --ops_type $ops_type --fuse_layer $fuse_layer --M $M --annotation_path $annotation_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/logs_$M$dataset$model${depth[$i]}$fuse_layer$ops_type$i.txt
done
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt
