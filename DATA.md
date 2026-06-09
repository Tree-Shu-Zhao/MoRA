
> **Tip:** Preprocessed, ready-to-train versions of all three datasets are available at
> [`TreezzZ/MoRA-data`](https://huggingface.co/datasets/TreezzZ/MoRA-data). See the
> "Download preprocessed data" option in the [README](README.md#-data-preparation) to skip the
> manual download/organize/preprocess steps below.

# Data directory structure:

## MM-IMDb
[MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb) [(archive.org mirror)](https://archive.org/download/mmimdb)

    root
    ├── images            
    │   ├── 00000005.jpeg 
    │   ├── 00000008.jpeg   
    │   └── ...        
    ├── labels          
    │   ├── 00000005.json 
    │   ├── 00000008.json   
    │   └── ...        
    └── split.json 


## Food101
[UPMC Food-101](https://visiir.isir.upmc.fr/explore) [(Kaggle)](https://www.kaggle.com/datasets/gianmarco96/upmcfood101?select=texts)

    root
    ├── images            
    │   ├── train                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    │   ├── test                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    ├── texts          
    │   ├── train_titles.csv            
    │   └── test_titles.csv         
    ├── class_idx.json         
    ├── text.json         
    └── split.json


##  Hateful Memes
[Hateful Memes](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/)

**Update:** The datasets we used are from [(Kaggle)](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset), however, the test.jsonl here does not contain the label information. To make the label available for the evaluation, we download the test_seen.jsonl from [(Kaggle2)](https://www.kaggle.com/datasets/williamberrios/hateful-memes), which has the same test set as the previous one with the label information. You can directly download the test_seen.jsonl file here and use it (of course you should rename it as test.jsonl or revise the write_*.py file).


    root
    ├── img           
    │   ├── xxxxx.png 
    │   ├── xxxxx.png   
    │   └── ...        
    ├── train.jsonl          
    ├── dev.jsonl           
    └── test.jsonl 
