#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

root_path='./'
annotation_path='./hmdb51_1.json'
video_path='../../../raid/video_data/hmdb51/frames'

model=sknet   #resnet  sknet... 
depth=(26 50 101)  # 50 101
M=2

batch_size=30

dataset=hmdb51
n_classes=51
sample_duration=30
checkpoint=200
n_epochs=400


i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --n_epochs $n_epochs --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint > logs/sknet_$M$dataset$model${depth[$i]}$i.txt
done
