#  Learn about Configs

## Use Pre-Trained Model
To use the pre-trained model for the whole network, the config adds the link of pre-trained models in the
`pretrain`. We set `pretrain=None` by default and use the CLIP model as the pre-trained model(if your `config.network.arch=ViT-B/32 or ViT-B/16` and`config.network.init = True` ).
You are able to do zero-shot validation pre-trained on Kinetics-400.     
```
pretrain: /home/ock/video_clip/video-clip/clip_models/k400_vit32_8frame.pt
```
To resume the interrupted model, the config adds the link `resume`. You are able to resume your training started from the last saved model.
```
resume: /home/ock/video_clip/video-clip/exp/clip_hmdb/ViT-B/32/hmdb/20210916_145953/3.pth.tar
``` 

## Modify Dataset
ActionCLIP supports  Kinetics-400, UCF101 and HMDB51 now. The users may need to adapt one of the above dataset to fit for their special datasets.
```
#dataset settings
data:
    dataset: hmdb                                                 #dataset names
    num_classes: 51                                               #dataset classes
    image_tmpl: 'img_{:05d}.jpg'                                  #Picture naming format
    train_list: 'lists/hmdb51/train_rgb_split1.txt'               #dataset traning list  
    val_list: 'lists/hmdb51/val_rgb_split1.txt'                   #dataset validation list
    label_list: 'lists/hmdb51_labels.csv'                         #dataset label list
```

## Modify Models
ActionCLIP supports ViT-B/32 and ViT-B/16 as the backbone. 
```
#model settings
network:
    arch: ViT-B/32  #ViT-B/32 ViT-B/16                          #Backbone
```
ActionCLIP supports `Pre-network Prompt: Joint`, `In-network Prompt: Shift`,  and
`Post-network Prompt:MeanP, Conv1D/LSTM, Transf/Transf_cls` now.
```
#visual prompt settings
network:
    tsm: False                                                  #Pre-network Prompt
    joint: False                                                #In-network Prompt
    sim_header: "seqTransf"                                     #Post-network Prompt(seqTransf meanP seqLSTM conv_1D seqTransf_cls)
```

## Modify Training Schedule
Finetuning usually requires smaller learning rate and less training epochs.
```
solver:
    # learning policy
    type: cosine       #cosine multistep
    epochs: 50
    start_epoch: 0
    epoch_offset: 0

    # optimizer 
    optim: adamw      #adam sgd adamw
    clip_gradient: 20
    loss_type: nll
    lr: 5.e-6
    lr_warmup_step: 5
    momentum: 0.9
    weight_decay: 0.0005
    lr_decay_step: 15
    lr_decay_factor: 0.1
```