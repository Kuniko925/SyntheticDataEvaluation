import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def reduce_dim(image_embeddings, df_train, reducer, root, model_name, reducer_name):
    embeddings_2d = reducer.fit_transform(image_embeddings)
    df_train['embeddings x'] = embeddings_2d[:,0]
    df_train['embeddings y'] = embeddings_2d[:,1]
    save_filepath = f'{root}distance/embed_{model_name}_{reducer_name}_train.csv'
    df_train.to_csv(save_filepath, index=False)
    return embeddings_2d

def get_dinov2_embeddings(images):
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

    #Extract the features
    image_embeddings = []
    with torch.no_grad():
        for img in tqdm(images):
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  
            image_embeddings.append(embedding.cpu().numpy().flatten())

    return image_embeddings

def get_dinov3_embeddings(images):
    #Load the model and processor
    pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(pretrained_model_name).to(device)
    
    #Extract the features
    image_embeddings = []
    with torch.no_grad():
        for img in tqdm(images):
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  
            image_embeddings.append(embedding.cpu().numpy().flatten())

    return image_embeddings

def get_clip_embeddings(images):

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_embeddings = []
    with torch.no_grad():
        for img in tqdm(images):
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            image_embeddings.append(outputs.cpu().numpy().flatten())

    return image_embeddings
    