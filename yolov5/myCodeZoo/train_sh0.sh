# train AUIF
python train.py --device 0 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-AUIF.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_AUIF --exist-ok
# train CDDFuse
python train.py --device 0 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-CDDFuse.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_CDDFuse --exist-ok
