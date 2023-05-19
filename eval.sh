
# OPEN_CLIP_HOME="/p/project/laionize/jitsev1_juwelsbooster/open_clip_lr_mod"
OPEN_CLIP_HOME="/admin/home-mitchellw/git/lm-evaluation-harness"
export PYTHONPATH="$PYTHONPATH:${OPEN_CLIP_HOME}"


cd /admin/home-mitchellw/git/lm-evaluation-harness/

model=$1
folder=$2
gpu=$3
etype=$4

while true
do
    for i in `ls -t /fsx/home-mitchellw/experimetns/lm/$folder/checkpoints/epoch*.pt`
    do

        #echo $model
        save_path="$(dirname $i)/eval_$(basename $i)"

        echo $save_path


        if [ -f "$save_path" ]; then
            echo "$save_path exists."
        elif [[ $save_path == *"latest"* ]]; then
            echo "pass on latest"
        else
            python main.py \
                --model $etype \
                --model_args pretrained_model=$model,path_to_checkpoint=$i \
                --tasks copa,winogrande,triviaqa,boolq,piqa,arc_easy,arc_challenge,hellaswag,lambada_openai \
                --device "cuda:$gpu" \
                --output_path $save_path
        fi
    done
    sleep 60
done


# bash eval.sh xlarge2048 200b-rpj-decayto3e-5-xlarge2048-10-1e-3-0.1-nodes16-v0 0 openlm
# bash eval.sh xlarge2048 200b-rpj-decayto3e-5-xlarge2048-10-2e-3-0.1-nodes16-v0 1 openlm
# bash eval.sh xlarge2048 200b-rpj-init-decayto3e-5-xlarge2048-10-2e-3-0.1-nodes16-v0 3 openlm
# bash eval.sh xlarge2048 200b-rpj-init-noreplace-decayto3e-5-xlarge2048-10-2e-3-0.1-nodes16-v0 4 openlm
# bash eval.sh m1b 200b-rpj-init-noreplace-decayto3e-5-m1b-10-2e-3-0.1-nodes16-v0 5 openlm
# bash eval.sh m1b 200b-rpj-100k-m1b-10-2e-3-0.1-nodes16-v0 6 openlm
# bash eval.sh m1b 200b-rpj-neox100k-m1b_neox-10-2e-3-0.1-nodes16-v0 7 openlm_neox




# python main.py --model openlm --model_args pretrained_model=xlarge2048,path_to_checkpoint=/fsx/home-mitchellw/experimetns/lm/200b-rpj-xlarge2048-10-5e-4-0.1-nodes16-v0/checkpoints/epoch_3.pt --tasks copa,winogrande,triviaqa,boolq,piqa,arc_easy,arc_challenge,hellaswag,lambada_openai --device cuda:1
# python main.py --model hf-causal  --model_args pretrained=mosaicml/mpt-1b-redpajama-200b --tasks copa,winogrande,triviaqa,boolq,piqa,arc_easy,arc_challenge,hellaswag,lambada_openai --device cuda:2 --output_path /fsx/home-mitchellw/mpt1bevals.pt

# exit
# ssh ip-26-0-129-105
# conda activate eval
# cd /admin/home-mitchellw/git/lm-evaluation-harness/