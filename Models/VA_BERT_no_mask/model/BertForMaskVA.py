from transformers.modeling_bert  import BertPreTrainedModel
from transformers.modeling_bert import BertModel
from transformers.modeling_bert import BertOnlyMLMHead
from transformers.modeling_bert import BertOnlyNSPHead

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

class BertVAHead(nn.Module):
    def __init__(self, config):
        super(BertVAHead, self).__init__()
        self.prediction_valence = nn.Linear(config.hidden_size, 1)
        self.prediction_arousal = nn.Linear(config.hidden_size, 1)

    def forward(self, pooled_output):
        valence_score = self.prediction_valence(pooled_output)
        valence_arousal = self.prediction_arousal(pooled_output)
        return valence_score, valence_arousal

class BertForMaskVA(BertPreTrainedModel):
    r"""

    """

    def __init__(self, config):
        super(BertForMaskVA, self).__init__(config)

        self.bert = BertModel(config)
        self.cls_MLM = BertOnlyMLMHead(config)
        self.cls_VA = BertVAHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls_MLM.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        label_V=None,
        label_A=None,
    ):
        #Output for MLM
        outputs_mlm = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs_mlm[0]
        prediction_scores = self.cls_MLM(sequence_output)

        outputs_mlm = (prediction_scores,) + outputs_mlm[2:]

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs_mlm = (masked_lm_loss,) + outputs_mlm

        
        #Output for VA
        outputs_VA = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs_VA[1]

        valence_score, arousal_score = self.cls_VA(pooled_output)

        outputs_VA = (valence_score, arousal_score) + outputs_VA[2:]  # add hidden states and attention if they are here  
        if label_V is not None:
            loss_fct = MSELoss()
            loss_V = loss_fct(valence_score, label_V)

        if label_A is not None:
            loss_A = loss_fct(arousal_score, label_A)
            
        outputs = masked_lm_loss+loss_V+loss_A
        outputs = (outputs,) + outputs_VA 
        
        return outputs  # (三個loss相加), outputs_VA[2:] -> [seq_relationship_score, (hidden_states), (attentions)]

"""
@add_start_docstrings(
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING
)
"""
