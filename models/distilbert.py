from transformers.models.distilbert import DistilBertModel
from torch import nn

class DistilBertWithWeights(nn.Module):
    def __init__(self, num_labels, weights=None):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss(weight=weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}
