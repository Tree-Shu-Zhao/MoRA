import json
import os

import hydra
import jsonlines
import pandas as pd
import pyarrow as pa
from tqdm import tqdm


def preprocess(cfg):
    if cfg.NAME.lower() == 'mmimdb':
        mmimdb_create_arrow_files(cfg.DATA_DIR)
    elif cfg.NAME.lower() == 'hatememes':
        hatememes_create_arrow_files(cfg.DATA_DIR)
    elif cfg.NAME.lower() == 'food101':
        food101_create_arrow_files(cfg.DATA_DIR)


def mmimdb_create_arrow_files(root):
    GENRE_CLASS = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror'
     , 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music',
     'Animation', 'Musical', 'Western', 'Sport', 'Short', 'Film-Noir']
    GENRE_CLASS_DICT = {}
    for idx, genre in enumerate(GENRE_CLASS):
        GENRE_CLASS_DICT[genre] = idx    

    image_root = os.path.join(root, 'images')
    label_root = os.path.join(root, 'labels')
    
    with open(f"{root}/split.json", "r") as fp:
        split_sets = json.load(fp)
        
    
    for split, samples in split_sets.items():
        data_list = []
        for sample in tqdm(samples):
            image_path = os.path.join(image_root, sample+'.jpeg')
            label_path = os.path.join(label_root, sample+'.json')
            with open(image_path, "rb") as fp:
                binary = fp.read()
            with open(label_path, "r") as fp:
                labels = json.load(fp)    
            
            plots = labels['plot']
                
            genres = labels['genres']
            label = [1 if g in genres else 0 for g in GENRE_CLASS_DICT]
            data = (binary, plots, label, genres, sample, split)
            data_list.append(data)

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "plots",
                "label",
                "genres",
                "image_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        assert split in ['train', 'dev', 'test', 'val', 'validation']
        if split in ['dev', 'val', 'validation']:
            split = 'val' # Unified naming
        with pa.OSFile(f"{root}/{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def hatememes_create_arrow_files(root):
    split_sets = ['train', 'dev', 'test']
    
    for split in split_sets:
        data_list = []
        with jsonlines.open(os.path.join(root,f'{split}.jsonl'), 'r') as rfd:
            for data in tqdm(rfd):
                image_path = os.path.join(root, data['img'])
                
                with open(image_path, "rb") as fp:
                    binary = fp.read()       
                    
                text = [data['text']]
                label = data['label']

                data = (binary, text, label, split)
                data_list.append(data)                
                            

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "text",
                "label",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        assert split in ['train', 'dev', 'test', 'val', 'validation']
        if split in ['dev', 'val', 'validation']:
            split = 'val' # Unified naming
        with pa.OSFile(f"{root}/{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

def food101_create_arrow_files(root):
    image_root = os.path.join(root, 'images')
    
    with open(f"{root}/class_idx.json", "r") as fp:
        FOOD_CLASS_DICT = json.load(fp)
        
    with open(f"{root}/text.json", "r") as fp:
        text_dir = json.load(fp)
        
    with open(f"{root}/split.json", "r") as fp:
        split_sets = json.load(fp)
        
    
    for split, samples in split_sets.items():
        split_type = 'train' if split != 'test' else 'test'
        data_list = []
        for sample in tqdm(samples):
            if sample not in text_dir:
                print("ignore no text data: ", sample)
                continue
            cls = sample[:sample.rindex('_')]
            label = FOOD_CLASS_DICT[cls]
            image_path = os.path.join(image_root, split_type, cls, sample)

            with open(image_path, "rb") as fp:
                binary = fp.read()
                
            text = [text_dir[sample]]
            
            
            data = (binary, text, label, sample, split)
            data_list.append(data)

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "text",
                "label",
                "image_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        assert split in ['train', 'dev', 'test', 'val', 'validation']
        if split in ['dev', 'val', 'validation']:
            split = 'val' # Unified naming
        with pa.OSFile(f"{root}/{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)        


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg):
    preprocess(cfg.dataset)


if __name__ == "__main__":
    main()