import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import os
import nltk
from tqdm import trange
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from seqeval.metrics import classification_report as seqeval_classification_report

# Download necessary NLTK data
nltk.download('punkt')

class NER_ClinicalBERT(object):
    # Define device to use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the tag indices and values as they were defined in the original model
    tag2idx = {'O': 0, 'ID': 1, 'PHI': 2, 'NAME': 3, 'CONTACT': 4, 'DATE': 5, 'AGE': 6, 'PROFESSION': 7, 'LOCATION': 8, 'PAD': 9}
    tag_values = ["O", "ID", "PHI", "NAME", "CONTACT", "DATE", "AGE", "PROFESSION", "LOCATION", "PAD"]

    # Initialize tokenizer and model with Clinical BERT
    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=False)
    model = BertForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=len(tag2idx))

    # Define constants for maximum sequence length and batch size
    MAX_LEN = 75  # Maximum length of the sequences
    bs = 32  # Batch size for training and evaluation

    def __init__(self):
        # Initialize lists to store loss values and training accuracies
        self.loss_values = []
        self.train_accuracies = []

        # Load the model state if it exists, otherwise use the pre-trained model
        if os.path.exists("Models/NER_ClinicalBERT.pt"):
            print("Loading model")
            state_dict = torch.load("Models/NER_ClinicalBERT.pt", map_location=self.device)
            print("Loaded model")
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print("Using pre-trained Clinical BERT model")
        
        # Move model to the device (GPU/CPU)
        self.model.to(self.device)

    def perform_NER(self, text):
        """
        Perform Named Entity Recognition (NER) on the given text using the Clinical BERT model.
        :param text: The input text for NER
        :return: A list of tuples containing tokens and their corresponding labels
        """
        # Tokenize text into sentences
        list_of_sents = sent_tokenize(text)
        list_of_tuples_by_sent = []

        # Loop through each sentence and perform NER
        for sent in list_of_sents:
            tokenized_sentence = self.tokenizer.encode(sent, truncation=True)
            input_ids = torch.tensor([tokenized_sentence]).to(self.device)

            # Predict using the model
            with torch.no_grad():
                output = self.model(input_ids)
            
            # Get label indices and convert tokens back to text
            label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

            # Process tokens and labels, handling subword tokens
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, label_indices[0]):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(self.tag_values[label_idx])
                    new_tokens.append(token)

            # Create list of tuples (token, label) for the sentence
            list_of_tuples = [(token, label) for token, label in zip(new_tokens, new_labels)]
            list_of_tuples_by_sent.append(list_of_tuples)

        return list_of_tuples_by_sent

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        """
        Tokenizes sentences while preserving the corresponding labels.
        :param sentence: List of words in the sentence
        :param text_labels: Corresponding labels for each word
        :return: Tokenized sentences and the extended list of labels
        """
        tokenized_sentence = []
        labels = []

        # Tokenize each word and extend the labels to match the number of subwords
        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Extend the tokenized sentence and labels list accordingly
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def transform_sequences(self, tokens_labels):
        """
        Transforms sequences of (token, label) into feature sequences.
        :param tokens_labels: List of tuples (token, label) for each sentence
        :return: input_ids and tags - the transformed sequences ready for model input
        """
        tokenized_sentences = []
        labels = []

        # Process each sentence and its labels
        for index, sentence in enumerate(tokens_labels):
            text_labels = []
            sentence_to_feed = []
            for word_label in sentence:
                text_labels.append(word_label[1])
                sentence_to_feed.append(word_label[0])
            
            # Tokenize sentences and preserve labels
            a, b = self.tokenize_and_preserve_labels(sentence_to_feed, text_labels)
            tokenized_sentences.append(a)
            labels.append(b)

        # Convert tokens to IDs and pad sequences
        input_ids = pad_sequences([NER_ClinicalBERT.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_sentences],
                                  maxlen=NER_ClinicalBERT.MAX_LEN, dtype="long", value=0.0,
                                  truncating="post", padding="post")

        # Convert labels to IDs and pad sequences
        tags = pad_sequences([[NER_ClinicalBERT.tag2idx.get(l) for l in lab] for lab in labels],
                             maxlen=NER_ClinicalBERT.MAX_LEN, value=NER_ClinicalBERT.tag2idx["PAD"], padding="post",
                             dtype="long", truncating="post")

        return input_ids, tags

    def learn(self, X_train, Y_train, epochs=1):
        """
        Train the Clinical BERT model on the provided training data.
        :param X_train: Input features (token IDs)
        :param Y_train: Corresponding labels
        :param epochs: Number of epochs to train the model
        """
        if torch.cuda.is_available():
            self.model.cuda()

        # Create masks and convert training data to tensors
        tr_masks = [[float(i != 0.0) for i in ii] for ii in X_train]
        tr_inputs = torch.tensor(X_train).type(torch.long).to(self.device)
        tr_tags = torch.tensor(Y_train).type(torch.long).to(self.device)
        tr_masks = torch.tensor(tr_masks).type(torch.long).to(self.device)

        # Create DataLoader for training
        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.bs)

        # Set up the optimizer and scheduler for fine-tuning
        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n in param_optimizer]}]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8
        )

        max_grad_norm = 1.0
        total_steps = len(train_dataloader) * epochs

        # Create scheduler for learning rate adjustment
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Training loop
        for _ in trange(epochs, desc="Epoch"):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_elements = 0

            # Iterate over batches
            for step, batch in enumerate(train_dataloader):
                batch = tuple(b.to(self.device) for b in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.model.zero_grad()

                # Forward pass
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                logits = outputs[1]

                # Backward pass and optimization
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Calculate accuracy for this batch
                predictions = torch.argmax(logits, dim=2)
                correct_predictions = (predictions == b_labels).sum().item()
                total_correct += correct_predictions
                total_elements += b_labels.numel()

            # Calculate and store average training loss and accuracy
            avg_train_loss = total_loss / len(train_dataloader)
            train_accuracy = total_correct / total_elements
            print("Average train loss: {}".format(avg_train_loss))
            print("Training Accuracy: {}".format(train_accuracy))
            self.loss_values.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)

            # Save model state after each epoch (optional)
            if not os.path.exists("Models/"):
                os.makedirs("Models/")
            torch.save(self.model.state_dict(), os.path.join("Models/", 'BERT_epoch-{}.pt'.format(_ + 1)))

        # Plot the learning curve after training
        plt.figure()
        plt.plot(self.loss_values, 'b-o', label="Training Loss")
        plt.plot(self.train_accuracies, 'g-o', label="Training Accuracy")  # Add training accuracy to the plot
        plt.title("Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig("training_loss_accuracy_curve.png")
        plt.show()

    def evaluate(self, X_test, Y_test):
        """
        Evaluate the trained Clinical BERT model on the test data.
        :param X_test: Input features (token IDs) for testing
        :param Y_test: Corresponding labels for testing
        """
        # Create masks and convert test data to tensors
        val_masks = [[float(i != 0.0) for i in ii] for ii in X_test]
        val_inputs = torch.tensor(X_test).type(torch.long)
        val_tags = torch.tensor(Y_test).type(torch.long)
        val_masks = torch.tensor(val_masks).type(torch.long)

        # Create DataLoader for validation
        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=NER_ClinicalBERT.bs)

        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        predictions, true_labels = [], []

        # Evaluation loop
        for batch in valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Forward pass without gradient calculation
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Accumulate evaluation loss
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        # Calculate average evaluation loss
        eval_loss = eval_loss / len(valid_dataloader)
        print("Validation loss: {}".format(eval_loss))

        # Convert predictions and true labels to tags
        pred_tags = [
            [NER_ClinicalBERT.tag_values[p_i] for p_i, l_i in zip(p, l) if NER_ClinicalBERT.tag_values[l_i] != "PAD"]
            for p, l in zip(predictions, true_labels)
        ]
        valid_tags = [
            [NER_ClinicalBERT.tag_values[l_i] for l_i in l if NER_ClinicalBERT.tag_values[l_i] != "PAD"]
            for l in true_labels
        ]

        # Calculate evaluation metrics
        accuracy = accuracy_score(valid_tags, pred_tags)
        f1 = f1_score(valid_tags, pred_tags, average='weighted')
        precision = precision_score(valid_tags, pred_tags, average='weighted')
        recall = recall_score(valid_tags, pred_tags, average='weighted')

        print("Validation Accuracy: {}".format(accuracy))
        print("Validation F1-Score: {}".format(f1))
        print("Validation Precision: {}".format(precision))
        print("Validation Recall: {}".format(recall))

        # Use seqeval's classification report for detailed results
        print(seqeval_classification_report(valid_tags, pred_tags, digits=4))

        # Plot the training and validation metrics
        if hasattr(self, 'loss_values'):
            epochs = list(range(1, len(self.loss_values) + 1))

            plt.figure(figsize=(12, 6))

            # Plot loss and training accuracy
            plt.subplot(1, 2, 1)
            plt.plot(epochs, self.loss_values, 'b-o', label="Training Loss")
            plt.plot(epochs, self.train_accuracies, 'g-o', label="Training Accuracy")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            
            # Plot validation accuracy and F1-score
            plt.subplot(1, 2, 2)
            plt.plot(epochs, [accuracy] * len(epochs), 'g-o', label="Validation Accuracy")
            plt.plot(epochs, [f1] * len(epochs), 'r-o', label="Validation F1-Score")
            plt.title("Validation Metrics")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()

            plt.tight_layout()
            plt.savefig("training_and_validation_metrics.png")
            plt.show()

    def save(self, model_path):
        """
        Save the trained model to disk.
        :param model_path: Name of the model file to be saved
        """
        torch.save(self.model.state_dict(), "Models/" + model_path + ".pt")
        print("Saved model to disk")
