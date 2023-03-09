start=`date +%s`
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 dist_train.py --part=1
python3 test.py --part=1
end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
