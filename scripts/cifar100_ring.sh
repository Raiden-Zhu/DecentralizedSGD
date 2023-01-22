
# Resnet18
python main.py --dataset_name "CIFAR100" --image_size 64 --batch_size 64 --mode "ring" --size 16  --lr 0.1 --model "ResNet18"  --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "CIFAR100" --image_size 64 --batch_size 512 --mode "ring" --size 16  --lr 0.8 --model "ResNet18"  --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0


# Resnet18_M
python main.py --dataset_name "CIFAR100" --image_size 56 --batch_size 64 --mode "ring" --size 16  --lr 0.1 --model "ResNet18_M"  --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "CIFAR100" --image_size 56 --batch_size 512 --mode "ring" --size 16  --lr 0.8 --model "ResNet18_M"  --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0


# AlexNet
python main.py --dataset_name "CIFAR100" --image_size 64 --batch_size 64 --mode "ring" --size 16  --lr 0.01 --model "AlexNet"  --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --device 0
python main.py --dataset_name "CIFAR100" --image_size 64 --batch_size 512 --mode "ring" --size 16  --lr 0.08 --model "AlexNet"  --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --device 0


# DenseNet
python main.py --dataset_name "CIFAR100" --image_size 64 --batch_size 64 --mode "ring" --size 16  --lr 0.1 --model "DenseNet121"  --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "CIFAR100" --image_size 64 --batch_size 512 --mode "ring" --size 16  --lr 0.8 --model "DenseNet121"  --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
