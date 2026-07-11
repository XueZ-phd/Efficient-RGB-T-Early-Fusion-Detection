# train TarDAL
python train.py --device 4 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-TarDAL.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_TarDAL --exist-ok
# train U2Fusion
python train.py --device 4 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-U2Fusion.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_U2Fusion --exist-ok
