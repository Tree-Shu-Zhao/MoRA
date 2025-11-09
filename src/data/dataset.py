import io
import os
import random

import pyarrow as pa
import torch
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import to_pil_image

Image.MAX_IMAGE_PIXELS = 1000000000


class MissingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        missing_params,
        dataset_name,
        split=None,
    ):
        super(MissingDataset, self).__init__()

        # Read data
        table = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{data_dir}/{split}.arrow", "r")).read_all()
        self.dataset_name = dataset_name
        
        self.images = table["image"]
        self.labels = table["label"]
        if dataset_name == "hatememes" or dataset_name == "food101":
            self.texts = table["text"]
        elif dataset_name == "mmimdb":
            self.texts = table["plots"]
        else:
            raise ValueError(f"dataset_name {dataset_name} not implemented.")
        assert len(self.images) == len(self.texts) == len(self.labels)
        
        # Index mapping
        # For mmimdb, each image has multiple plots, so we need to map the index to the original image and plot index
        self.index_mapping = []
        for i in range(len(self.images)):
            texts = self.texts[i].as_py()
            for j in range(len(texts)):
                self.index_mapping.append((i, j))

        # Missing modality control        
        missing_ratio = missing_params.RATIO
        missing_type = missing_params.TYPE
        both_ratio = missing_params.BOTH_RATIO
        missing_table_root = missing_params.SAVE_ROOT
        os.makedirs(missing_table_root, exist_ok=True)
        missing_table_name = f'{dataset_name}_split{split}_missing_{missing_type}_{missing_ratio}.pt'
        missing_table_path = os.path.join(missing_table_root, missing_table_name)
        
        total_num = len(self.images)
        
        # Create or load a missing table
        if os.path.exists(missing_table_path):
            missing_table = torch.load(missing_table_path)
            if len(missing_table) != total_num:
                logger.error('missing table mismatched!')
                exit()
        else:
            missing_table = torch.zeros(total_num)
            
            if missing_ratio > 0:
                missing_index = random.sample(range(total_num), int(total_num*missing_ratio))

                if missing_type == 'text':
                    missing_table[missing_index] = 1
                elif missing_type == 'image':
                    missing_table[missing_index] = 2
                elif missing_type == 'both':
                    missing_table[missing_index] = 1
                    missing_index_image = random.sample(missing_index, int(len(missing_index)*both_ratio))
                    missing_table[missing_index_image] = 2
                    
                torch.save(missing_table, missing_table_path)

        self.missing_table = missing_table
        
    def __getitem__(self, index):
        image_index, text_index = self.index_mapping[index]
        image = self.images[image_index].as_py()
        text = self.texts[image_index].as_py()[text_index] # For mmimdb, each image has multiple plots, so we need to map the index to the original image and plot index
        label = self.labels[image_index].as_py()
        missing_type = self.missing_table[image_index].item()

        image = Image.open(io.BytesIO(image)).convert("RGB")
        
        # missing image, dummy image is all-one image
        if missing_type == 2:
            image = to_pil_image(torch.ones(image.size)).convert("RGB")
            
        # missing text, dummy text is 'None'
        if missing_type == 1:
            text = ''
        
        return {
            "image": image,
            "text": text,
            "label": label,
            "missing_type": missing_type,
        }

    def __len__(self):
        return len(self.index_mapping)


def collate_fn(batch, processor):
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]
    labels = [b["label"] for b in batch]
    missing_types = [b["missing_type"] for b in batch]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    labels = torch.tensor(labels)
    if labels.dim() == 2:
        labels = labels.float() # For multi-label classification
    missing_types = torch.tensor(missing_types).long()

    # If the input_ids length is maximum (77), after appending a vision token, it may exceed the maximum length and the eos token will be removed.
    # To prevent this, we replace the second last token with the pad token.
    for input_id in inputs.input_ids:
        if input_id[-2] != 49407: # EOS token
            input_id[-2] = processor.tokenizer.pad_token_id

    return {
        "inputs": inputs,
        "labels": labels,
        "missing_types": missing_types,
    }