# this script is for testing kinetics+Moment datasets

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

pre_path=' '  #'--pretrain_path results/mit_sknet26_M2__save_6.pth --n_pretrain_classes 339'
resume_path=' --resume_path results/hollywood2_sknet326_M4__save_200.pth '
root_path='./'
annotation_path='./'
video_path='../../../raid/Hollywood2'

model=sknet3   #resnet  sknet... 
depth=(26) # 50 101) 
M=4

batch_size=32

dataset=hollywood2
n_classes=30
sample_duration=30
checkpoint=100
epoch=200

i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=7 python main.py $resume_path --inference_batch_size 25 --no_train --no_val --inference --inference_subset test --n_epochs $epoch --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint #> logs/sknet_$M$dataset$model${depth[$i]}$i.txt
done
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt