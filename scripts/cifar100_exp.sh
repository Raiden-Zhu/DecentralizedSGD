

# Resnet18_M

python main.py --dataset_name "CIFAR100" --image_size 56 --batch_size 64 --mode "exponential" --size 16  --lr 0.2 --model "ResNet18_M"  --warmup_step 300 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "CIFAR100" --image_size 56 --batch_size 512 --mode "exponential" --size 16  --lr 1.6 --model "ResNet18_M"  --warmup_step 300 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
