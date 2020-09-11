#TEST 0

#sudo nvidia-docker run --shm-size 32G --rm --name sknet -v $(pwd):/workspace -it 3dresnet
#cd home/sghong/SKNet-PyTorch
#bash exp_scripts/debug.sh

#pre_path=(' ', '--pretrain_path pretrained/r3d50_KMS_200ep.pth --n_pretrain_classes 1139' '--pretrain_path pretrained/r3d101_KM_200ep.pth --n_pretrain_classes 1039' '--pretrain_path pretrained/r3d200_KM_200ep.pth --n_pretrain_classes 1039' '--pretrain_path pretrained/r3d152_KM_200ep.pth --n_pretrain_classes 1039')

root_path='./'
annotation_path='./ucf101_01.json'
video_path='../../../raid/video_data/ucf101/ucf101_videos'

model=sknet   #resnet  #3dresnet, 3dsknet ... 
depth=(50 50 101 200 152)
batch_size=32  #Out of memory when 32

dataset=ucf101
n_classes=101
sample_duration=30

i=0
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes --ft_begin_module fc --model $model --model_depth 50 --batch_size $batch_size --n_threads 4 --checkpoint 200 > logs/sknet_tmp$i.txt

#for i in "${!pre_path[@]}"; do
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --root_path $root_path --annotation_path $annotation_path --video_path $video_path --sample_duration $sample_duration --result_path results  --dataset $dataset --n_classes $n_classes ${pre_path[$i]} --ft_begin_module fc --model resnet --model_depth ${depth[$i]} --batch_size $batch_size --n_threads 4 --checkpoint 5 #> jig_tmp$i.txt

#done
