import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import (
    WEIGHTS_NAME, BertConfig, BertModel, BertPreTrainedModel, BertTokenizer)
from torch.nn import MSELoss, CrossEntropyLoss


def l2_loss(parameters):
    return torch.sum(
        torch.tensor([
            torch.sum(p ** 2) / 2 for p in parameters if p.requires_grad
        ]))


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode(
            "Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.l2_reg_lambda = config.l2_reg_lambda
        self.bert = BertModel(config)
        self.latent_entity_typing = config.latent_entity_typing
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_size = config.hidden_size*3
        if self.latent_entity_typing:
            classifier_size += config.hidden_size*2
        self.classifier = nn.Linear(
            classifier_size, self.config.num_labels)
        self.latent_size = config.hidden_size
        self.latent_type = nn.Parameter(torch.FloatTensor(
            3, config.hidden_size), requires_grad=True)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, e1_mask=None, e2_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # for details, see the document of pytorch-transformer
        pooled_output = outputs[1]
        sequence_output = outputs[0]
        #pooled_output = self.dropout(pooled_output)

        def extract_entity(sequence_output, e_mask):
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(
                extended_e_mask.float(), sequence_output).squeeze(1)
            return extended_e_mask.float()
        e1_h = extract_entity(sequence_output, e1_mask)
        e2_h = extract_entity(sequence_output, e2_mask)
        context = self.dropout(pooled_output)
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)

    #
    #   second_pre = - tf.reduce_max(rc_probabilities[:, 1:], axis=-1) + 1
    #   rc_loss = - tf.math.log(second_pre)#+ tf.math.log(second_pre) * log_probs[:,0]
        # print(pooled_output.size())
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        device = logits.get_device()
        l2 = l2_loss(self.parameters())
        # print(l2)
        if device >= 0:
            l2 = l2.to(device)
        loss = l2 * self.l2_reg_lambda
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss += loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                # loss += loss_fct(
                #     logits.view(-1, self.num_labels), labels.view(-1))
                # I thought that using Gumbel softmax should be better than the following code.

                probabilities = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                one_hot_labels = F.one_hot(labels, num_classes=self.num_labels)
                if device >= 0:
                    one_hot_labels = one_hot_labels.to(device)

                dist = one_hot_labels[:, 1:].float() * log_probs[:, 1:]
                example_loss_except_other, _ = dist.min(dim=-1)
                per_example_loss = - example_loss_except_other.mean()

                rc_probabilities = probabilities - probabilities * one_hot_labels.float()
                second_pre,  _ = rc_probabilities[:, 1:].max(dim=-1)
                rc_loss = - (1 - second_pre).log().mean()

                #print(loss, per_example_loss, rc_loss)
                loss += per_example_loss + 5 * rc_loss

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
