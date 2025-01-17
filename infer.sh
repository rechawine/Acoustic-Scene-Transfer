
# --desc_prompt "She is talking in a large room." \
# --desc_prompt "clean speech"

# python generate.py \
#     --ckpt_path /ssd6/other/liangzq02/code2/VoiceLDM/pretrain/voiceldm-m.ckpt \
#     --desc_prompt "She is talking in a very large room." \
#     --cont_prompt "Talkin' 'bout the days when we used to ride in style, Rockin' all the bling with that icy cold smile" \
#     --desc_guidance_scale 2 --cont_guidance_scale 8


python generate.py \
    --ckpt_path /ssd6/other/liangzq02/code2/VoiceLDM/results/VoiceLDM-M/checkpoints/checkpoint_1/pytorch_model.bin \
    --desc_prompt "/ssd6/other/liangzq02/data/RIR_44k/example/with_reverb.wav" \
    --cont_prompt "/ssd5/other/zhangry04/分离数据切片/去除静音/oss数据/train_cut/vocal/__5nzrx-9bg/vocal_30.wav" \
    --desc_guidance_scale 2 --cont_guidance_scale 8