#TEST 0

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

pre_path=' '  #'--pretrain_path results/mit_sknet26_M2__save_6.pth --n_pretrain_classes 339'
resume_path=' --resume_path results/ucf101_sknet250_M4__save_200.pth '
#ucf101_sknet226_M4__save_200.pth
root_path='./'
annotation_path='./ucf101_01.json'
video_path='../../../raid/video_data/ucf101/ucf101_videos'


model=sknet2   #resnet  sknet... 
depth=(50) # 
M=4

batch_size=16

dataset=ucf101
n_classes=101
sample_duration=30
checkpoint=200
epoch=200

n_epochs=200
learning_rate=0.1
#inference=' --inference '


i=0
for i in "${!depth[@]}"; do
    CUDA_VISIBLE_DEVICES=7 python main_attn.py --no_train $resume_path $pre_path --learning_rate $learning_rate --n_epochs $n_epochs --M $M --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint $checkpoint #> logs/attention_$M$dataset$model${depth[$i]}$i.txt
done
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt
