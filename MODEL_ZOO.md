# ActionCLIP Model Zoo

## Kinetics
We experiment ActionCLIP with different backbones(we choose Transf as our final visual
prompt since it obtains the best results) and input frames configurations on k400. Here is the list of pre-trained models that we provide (see Table 6 of the paper).

| model             | n-frame     | top1 Acc(single-crop) | top5 Acc(single-crop)| checkpoint                                                   |
| :-----------------: | :-----------: | :-------------: |:-------------: |:---------------------------------------------------------: | 
|ViT-B/32 | 8 | 78.36%          | 94.25%|[link]() 
| ViT-B/16  | 8 |   81.09%    | 95.49% |[link]() 
| ViT-B/16 | 16 | 81.68%  | 95.87% |[link]() 
| ViT-B/16 | 32 |82.32%    | 96.20% |[link]()                                                       

## UCF101&&HMDB51
On HMDB51 and UCF101 datasets, the accuracy(k400 pretrained) is reported under the accurate setting.

### UCF101

| model             | n-frame     | top1 Acc(single-crop) | checkpoint                                                   |
| :-----------------: | :-----------: | :-------------: |:---------------------------------------------------------: | 
|ViT-B/16 | 32 | 97.1%          | [link]() 

### HMDB51
| model             | n-frame     | top1 Acc(single-crop) | checkpoint                                                   |
| :-----------------: | :-----------: | :-------------: |:---------------------------------------------------------: | 
|ViT-B/16 | 32 | 76.2%          | [link]() 
