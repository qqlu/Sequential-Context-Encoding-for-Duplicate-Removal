export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-8.0/lib64

NUM=2
for i in ${NUM}
do
	srun --partition=bj11part --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=RNN_NMS python train_model.py --num_epochs=120 --step=100 --output_dir=/mnt/lustre/liushu1/qilu_ex/RNN_NMS/final/ --learning_rate=0.01 --load=0 
done