# train RFNNest
python train.py --device 3 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-RFNNest.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_RFNNest --exist-ok
# train SeAFusion
python train.py --device 3 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-SeAFusion.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_SeAFusion --exist-ok
