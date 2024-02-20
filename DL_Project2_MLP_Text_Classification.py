#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install portalocker')
get_ipython().system('pip install torchmetrics')


# In[ ]:


pip install torchtext


# In[ ]:


import argparse
import logging
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchtext
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.datasets import DATASETS
from torchtext.utils import download_from_url
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchmetrics

_FILL_ = '_FILL_'
SEED = 1


# Set up the optimization problem where we take a random y of data and want theta to converge to this y.

# In[ ]:


from torch._functorch.vmap import lazy_load_decompositions
# Short Question

torch.manual_seed(SEED)

# Define y to be a target of dimension (1, 3) without a gradient
y = torch.tensor([[0.9, 0.5, 0.3]], requires_grad=False)

# Define theta to be a random tensor of dimension (1, 3) which requires a gradient; we want theta to converge to y
theta = torch.randn(1, 3, requires_grad=True)
print(theta)

# Define an SGD optimizer with learning rate 0.01 which acts on theta
optimizer = torch.optim.SGD([theta], lr=0.01)

# Fil in the code below using the optimizer above to get theta to converge to y
for epoch in range(100):
  # Zero out the gradients of l with respect to theta
  optimizer.zero_grad()

  # Define a loss manually which is ||theta-x||_{2}^{2}, the L2 loss across all components
  loss = torch.norm(theta - y, p = 2)

  print('Epoch:{} Loss: {}'.format(epoch, loss))

  # Get teh gradients of l with respect to theta
  loss.backward()

  # Update theta
  optimizer.step()

# These should look very similar
print(y)
print(theta)
with torch.no_grad():
  # Check the y and theta have converged to almost the same thing
  loss = torch.norm(theta - y, p = 2)
  assert (loss.item() - 0.0)**2 <= 0.001


# ### Note that if we don't implement optimizer.zero_grad():
# ### The gradient information would cumulate over time, and optimizer.zero_grad() would help us zero out the gradients such that for the next epoch, we wouldn't carry any unnecessary previous gradient information.
# ### If we only call optimizer.zero_grad() every 3 batches, for every 2nd (and 3rd) batch, we were essentially including gradient information from previous 1 (and 2) batche(s), so we make larger steps than we were supposed to.

# # Neural Text Classifier - Information
# 
# We will build a basic Neural Text Classifier. 
# 
# 
# The at a high level, the idea of this model goes as follows.
# - We are given a training set $\{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$ where each $x^{(i)}$ is a sentence and $y^{(i)}$ is a class label.
# - First, we need to loop over $\{x^{(i)}\}_{i=1}^{N}$ and get the Vocabulary, the number of unique words we see.
# - Once we do this, we will express each word as a one-hot representation. To do this, we will use a mapping from a unique word to an integer. For example, "the" might get index 3 and if there are 10 words (in the entire Vocabulary) then "the" would have a vector representation $x_{the} = (0,0,1,0,0,0,0,0,0,0)$. There will be many words in this Vocabulary, over 13,000. For this example, each word is mapped to a unique integer.
# - We will feed batches of data to the model and each batch will be transformed into a tensor with words each word transformed to its integer index in VOCAB below.
# - For example, we might get [["the man walks"], ["this is a sentence"]] -> [["the", "man", "walks"], ["this", "is", "a", "sentence"]] -> [[1, 4, 5], [6, 7, 8, 15]]. It depends on what unique integer each word gets.
# - Different sentences have different numbers of tokens but all batches need to be the same dimension (this is how PyTorch works), so we need a padding token. So, for example, if the batch size is B = 2 and we given two sentences like ["a b c", "a b c d e"] then as a tensor this will become [[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]] and notice that we padded the first example so that the tensor is of dimension (2, 5) with M = 5. In some sense, in each batch we need to figure out the maximum number of tokens for an instance and pad each instance to have the same length as this longest instance. To do the above, use the [collate function](https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders). The idea here is that the Dataloader takes in raw data and the collate function is applied to this data, returning formatting tensors we can use later on in the optimization. You'll fill this in, using the hints.  
# - After padding, we feed batches of data to the classifier, these are of dimension (B, M). For example, we have a batch size of 2 above and M = 5. This will depend on the batch but here the batch size is B.
# - Once we feed in (B, M) data to the network, we rewrite this as (B, M, vocab_size) by using a one-hot representation for each word.
# - Then, we do as it hints in the model's forward method. We first take an average agross all the M elements of each element of the batch to get a (B, vocab_size) tensor that represents each instance. We pass this tensor through linear layer and nonlinear layers as unusual. The model returns logits, without the Softmax applied. This is a multiclass classfication task.
# 
# Finally, we optimize the network and check it's train and validation set accuracies. We'll use both direct methods and torchmetrics to do this.

# ### Information (if interested in more)
# - torchtext repo: https://github.com/pytorch/text/tree/main/torchtext
# - torchtext documentation: https://pytorch.org/text/stable/index.html
# - collate function: https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
# - embedding layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

# ### Constants

# In[ ]:


# This is the dataset we will use
DATASET = "AG_NEWS"
DATA_DIR = ".data"
# We will just use CPU here, but if you have time try "cuda"
DEVICE = "cpu"
LR = 8.0
BATCH_SIZE = 16
NUM_EPOCHS = 5
MIN_FREQUENCY = 20
# Padding valued used; if we have a tensor data x = [[1,2,3], [4, 5], [1,2,3,4,5]] this needs padding
# As a tensor, this is t = [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [1, 2, 3, 4, 5]]
PADDING_VALUE = 0
PADDING_IDX = PADDING_VALUE

SEED = 1


# # Get the tokenizer

# In[ ]:


# A basic tokenizer by using get_tokenizer; pass "basic_english"
basic_english_tokenizer = get_tokenizer("basic_english")


# In[ ]:


basic_english_tokenizer("This is some text ...")


# In[ ]:


# Save the tokenizer as a contant; this is needed later
TOKENIZER = basic_english_tokenizer


# ### Get the data and get the vocabulary.

# In[ ]:


# Loop through all the (label, text) data and yield a tokenized version of text
def yield_tokens(data_iter):
    tokens = []
    for _, text in data_iter:
        tokens.append(TOKENIZER(text))
    return tokens


# In[ ]:


train_iter = DATASETS[DATASET](root=DATA_DIR, split="train")


# In[ ]:


# Use build_vocab_from_iterator to get the the vocabulary
# This is essentially a dictionary going from a word to a unique integer
# Make sure to specify the specials
VOCAB = build_vocab_from_iterator(
    yield_tokens(train_iter),
    min_freq = MIN_FREQUENCY,
    specials=('<pad>', '<unk>')
)

# Set the default index to 1
# Otherwise, VOCAB['unknownbigword'] will raise an Exception
# I.e. we want '<unk>' to be the unknown word
VOCAB.set_default_index(VOCAB['<unk>'])


# In[ ]:


assert VOCAB['<unk>'] == 1


# Examples

# In[ ]:


VOCAB['yoyooyoyoy'], VOCAB['house'], VOCAB['<pad>'], VOCAB['<unk>']


# In[ ]:


print(len(VOCAB))

# stoi = VOCAB.get_stoi()
# print(len(stoi))
# # print(len(set(yield_tokens(train_iter))))
# yield_tokens(train_iter)[0]


# In[ ]:


VOCAB(TOKENIZER("House house houses ThisisnotaKNownWord"))


# ### Helper functions

# In[ ]:


from torchtext.vocab.vocab_factory import Vocab

# Utility to transform text into a list of ints
# This shoould go "a b c" -> ["a", "b", "c"] -> [1, 2, 3], for example
def text_pipeline(x):
    # Apply tokenizer to x
    tokens = TOKENIZER(x)


    # Return the Vocab at those tokens

    l = []
    for token in tokens:

      l.append(VOCAB[token])

    return l

# Return a 0 starting version of x
# If x = "1" this should return 0
# If x = "3" this should return 2, Etc.
def label_pipeline(x):
    return int(x) -1


# Nice link on collate_fn and DataLoader in PyTorch: https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3

# In[ ]:



# For a batch of data that might not be a tensor, return the batch in ternsor version
# batch is a length B lsit of tuples where each element is (label, text)
# label is a raw string like "1" here; text is a sentence like "this is about soccer"
def collate_batch(batch):
    label_list, text_list = [], []
    for (label, text) in batch:
        # Get the label from {1, 2, 3, 4} to {0, 1, 2, 3} and append it to label list
        label_list.append(label_pipeline(label))

        # Return a list of ints
        processed_text = torch.tensor(text_pipeline(text))
        text_list.append(processed_text)

    # Make label_list into a tensor of dtype=torch.int64
    label_list = torch.tensor(label_list, dtype=torch.int64)

    # Pad the sequence
    # For Exmaple: if we had 2 elements and [[1, 2], [1,2,3,4]] in the text_list then we want
    # to have [[1, 2, 0, 0], [1, 2, 3, 4]] in text_list and text_list is a tensor
    # Look up pad_sequence and make sure you specify batch_first=True and specify the padding_value=0
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)

    # Return the data and put it on a GPU or CPU, as needed
    return label_list.to(DEVICE), text_list.to(DEVICE)


# ### Get the data

# In[ ]:


# Get an iterator for the AG_NEWS dataset and get the train version
train_iter = DATASETS[DATASET](root=DATA_DIR, split="train")

# Use the above to get the number of class elements
tup = [label for label, text in train_iter]
num_class = len(set(tup))
# What are the classes?
print(f"The number of classes is {num_class} ...")


# ### Set up the model

# In[ ]:


import numpy
# A very naive model used to classify text
class OneHotTextClassificationModel(nn.Module):
    def __init__(self, vocab_size, num_class):
        super(OneHotTextClassificationModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_class = num_class

        # Have this layer take in data of dimension vocab_size and return data of dimension 100
        self.fc1 =  nn.Linear(vocab_size, 100, bias = False)

        # We will not use this, but see below as we want to mimic this layer using one_hot and fc1
        self.e = nn.Embedding(vocab_size, 100)

        # Have this layer take in 100 and return data of dimension num_class
        self.fc2 = nn.Linear(100, num_class, bias = False)
        self.init_weights()

        # See forward below; we do not use this but you can use this if you want to to check
        self.use_embedding_layer = False

    def init_weights(self):
        # Initialize the weights of fc1 to the same exact data as what self.e has
        # Initialize the bias to zero
        self.fc1.weight.data = numpy.transpose(self.e.weight.data)

        # Unitialize fc2 to uniform between -0.5 and 0.5
        initrange = -0.5
        self.fc2.weight.data.uniform_(initrange,-initrange)

    def forward(self, x):
        B, K = x.shape
        # x is of dimension (B, K), where K is the maximum number of tokens in an element of the batch
        if not self.use_embedding_layer:
          # Transform x to a tensor where each element is one-hot encoded
          x = F.one_hot(x,num_classes = self.vocab_size).float() ### to float, otherwise error in later codes
          assert(x.shape == (B, K, self.vocab_size))

          # Pass x through fc1 to get the row in fc1 correspondng to the row x is
          x = self.fc1(x)
          assert(x.shape == (B, K, 100))
        else:
          # Note: the above two steps should be the same as doing the command below
          x = self.e(x)
          assert(x.shape == (B, K, 100))

        # Take the mean of the embedings for all words in each sentence
        x = x.mean(dim = 1)
        assert(x.shape == (B, 100))

        # Apply ReLU to x
        x = F.relu(x)
        assert(x.shape == (B, 100))

        # Pass through fc2
        x = self.fc2(x)
        assert(x.shape == (B, self.num_class))

        # Return the Logits
        return x


# In[ ]:


torch.manual_seed(SEED)


# ### Set up the data

# In[ ]:


# Map the data to the right format
train_iter, test_iter = DATASETS[DATASET]()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

# Split data into train and validation
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# Set up different DataLoaders
train_dataloader = DataLoader(split_train_, batch_size = BATCH_SIZE, shuffle = True, collate_fn= collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size = BATCH_SIZE, shuffle = True, collate_fn= collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn= collate_batch)


# In[ ]:





# ### Train the model

# In[ ]:


def train(dataloader, model, optimizer, criterion, epoch):
    # Put the model in train mode; this does not matter right now
    _FILL_
    total_acc, total_count = 0, 0
    total_loss = 0.0
    log_interval = 200

    for idx, (label, text) in enumerate(dataloader):
        # Zero out the gradients
        optimizer.zero_grad()


        # Get the predictions
        predicted_label = model(text)

        # Get the loss.
        loss = loss_fn(input=predicted_label, target=label)

        # The loss is computed by taking a mean, get the sum of the terms on the numerator
        with torch.no_grad():
          total_loss += loss.item() * len(label)

        # Do back propagation
        loss.backward()

        # Clip the gradients to have max norm 0.1
        # Look up torch.nn.utils.clip_grad_norm
        torch.nn.utils.clip_grad_norm(model.parameters(),0.1)

        # Do an optimization step.
        optimizer.step()

        # Get the accuracy
        # predicted_label is (B, num_class) so take the argmax over the right dimension to get the actual label
        total_acc += (predicted_label.argmax(dim = 1) == label).sum().item()

        # Update the total number of items
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f} "
                "| loss {:8.3f}".format(
                    epoch, idx,
                    len(dataloader),
                    total_acc / total_count,
                    total_loss / total_count
                    )
            )
            total_acc, total_count, total_loss = 0, 0, 0.0


for epoch in range(100):
  # Zero out the gradients of l with respect to theta
  optimizer.zero_grad()

  # Define a loss manually which is ||theta-x||_{2}^{2}, the L2 loss across all components
  loss = torch.norm(theta - y, p = 2)

  print('Epoch:{} Loss: {}'.format(epoch, loss))

  # Get teh gradients of l with respect to theta
  loss.backward()

  # Update theta
  optimizer.step()

# These should look very similar
print(y)
print(theta)
with torch.no_grad():
  # Check the y and theta have converged to almost the same thing
  loss = torch.norm(theta - y, p = 2)
  assert (loss.item() - 0.0)**2 <= 0.001


# In[ ]:


from torchmetrics.classification import Accuracy

def evaluate(dataloader, model):
    # Put the model in eval model; this does not matter right now
    model.eval()


    # Set this to Accuracy from torchmetrics; use multiclass and specify the number of labels
    accuracy_fn = Accuracy(task = 'multiclass',num_classes=num_class)

    total_acc = 0.0
    total_count = 0.0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            # Get the predictions
            predicted_label = model(text)
            # Get the number of samples we have, the denominator of accuracy
            total_count += label.size(0) ##

            # Get the total number of times we have the correct predictions, use accuracy_fn
            total_acc += accuracy_fn(predicted_label, label).item() * len(label)

            # Use accuracy_fn from torchmetrics to check that the total number of correct predictions is the same as if you use argmax on predicted_label
            assert (
                accuracy_fn(predicted_label,label).item() * len(label) == (predicted_label.argmax(dim = 1) == label).sum().item()
            )


    accuracy = total_acc/total_count
    return accuracy


# # Train the model
# 
# We should get an accuracy > 80% for the training set. This might take quite a bit of time to run since we use one-hot.

# In[ ]:


# Set up the loss function
# Note that this should be a multiclass classification problem and you take in logits
loss_fn = nn.CrossEntropyLoss().to(DEVICE)

# Instantiate the model
# Pass in the number of elements in VOCAB and num_class
model = OneHotTextClassificationModel(len(VOCAB), num_class).to(DEVICE)

# Instantiate the SGD optimizer with parameters LR
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


# In[ ]:


for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader, model, optimizer, loss_fn, epoch)
    accu_val = evaluate(valid_dataloader, model)
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch,
            time.time() - epoch_start_time,
            accu_val
            )
    )
    print("-" * 59)

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader, model)
print("test accuracy {:8.3f}".format(accu_test))


# In[ ]:




