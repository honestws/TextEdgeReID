import clip
import torch.nn.functional as F
from torch import nn

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=256,
            dropout=0.1
    ):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(self, CFG):
        super(CLIPModel, self).__init__()
        model, preprocess = clip.load('ViT-L/14')
        self.model = model
        self.preprocess = preprocess
        self.image_projection = \
            ProjectionHead(embedding_dim=CFG.image_embedding,
                           projection_dim=CFG.projection_dim,
                           dropout=CFG.dropout)
        self.text_projection = \
            ProjectionHead(embedding_dim=CFG.text_embedding,
                           projection_dim=CFG.projection_dim,
                           dropout=CFG.dropout)
        self.temperature = CFG.temperature

    def forward(self, imgs, txts):
        image_features = self.model.encode_image(imgs)
        text_features = self.model.encode_text(txts)
        # image_features_ = self.model.visual(imgs)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()
