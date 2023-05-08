# pip install ultralytics

DATA_FOLDER=/mnt/hd1/data/manuscripts
start : gen2

test :
	MINEKOLEVEL=INFO python assemble.py --data_folder=$(DATA_FOLDER)

gen :
	PYTHONPATH=../handwriting-synthesis MINEKOLEVEL=INFO python gen.py --data_folder=$(DATA_FOLDER)

gen2 :
	CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../pytorch-handwriting-synthesis-toolkit MINEKOLEVEL=INFO python gen2.py --frm=130

train :
	CUDA_VISIBLE_DEVICES=0  MINEKOLEVEL=INFO python train.py


