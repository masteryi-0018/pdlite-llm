for i in $(seq 0 2)
do
  python onnx2paddle.py $i
done

python onnx2paddle.py -1
python quantize.py
mkdir chatglm2-6b-opt
python opt.py

echo "__________convert finished__________"