start=`date +%s`
python3 train.py --device="cuda:1" --part=1
end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"

