from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import random
import code


class BertForCoQA(BertPreTrainedModel):
    def __init__(
            self,
            config,
            output_attentions=False,
            keep_multihead_output=False,
            cls_alpha=1.0,
            mask_p=0.0,
    ):
        super(BertForCoQA, self).__init__(config)
        self.cls_alpha = cls_alpha
        self.mask_p = mask_p
        self.bert = BertModel(
            config,
            # output_attentions=output_attentions,
            # keep_multihead_output=keep_multihead_output,
        )
        self.qa_outputs = nn.Linear(config.hidden_size, 2) # 进行两个位置的输出
        # self.cls_outputs_mid = nn.Linear(config.hidden_size,
        #                                  config.hidden_size)
        self.output_attentions = False
        self.cls_outputs = nn.Linear(config.hidden_size, 4) # 进行句子类别的输出
        self.apply(self.init_bert_weights)
        print("\n BertForCoQA \n")

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            start_positions=None,
            end_positions=None,
            cls_idx=None,
            # head_mask=None,
    ):

        outputs = self.bert( 
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            # head_mask=head_mask,
        ) # (encoded_layers[2, 384, 768],  pooled_output[2, 768]) ,  batch_size,seq,hidden_size   batch_size,hidden_size
        if self.output_attentions:
            all_attentions, sequence_output, cls_outputs = outputs
        else:
            sequence_output, cls_outputs = outputs
        span_logits = self.qa_outputs(sequence_output) # ([384,2],[384,2])
        # cls_outputs = self.dropout(cls_outputs)
        cls_logits = self.cls_outputs(cls_outputs) # [2,4]
        # guofeng

        start_logits, end_logits = span_logits.split(1, dim=-1) # [2,384],[2,384]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            cls_loss_fct = CrossEntropyLoss()
            start_loss = span_loss_fct(start_logits, start_positions)
            end_loss = span_loss_fct(end_logits, end_positions)
            cls_loss = cls_loss_fct(cls_logits, cls_idx)
            total_loss = (start_loss +
                          end_loss) / 2 + self.cls_alpha * cls_loss
            return total_loss
        elif self.output_attentions:
            return all_attentions, start_logits, end_logits, cls_logits

        return start_logits, end_logits, cls_logits
