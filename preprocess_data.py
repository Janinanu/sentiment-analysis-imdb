#%%
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoModel


#%% 
# Load the IMDb dataset
dataset = load_dataset("imdb")

#%%
# Inspect the first sample in the training set
print(len(dataset['train'][:1]["text"][0].split()))

# %%
# Load the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define a preprocessing function
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# Apply the preprocessing function to the dataset
tokenized_data = dataset.map(preprocess, batched=True)

# %%
# Set format for PyTorch
tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Create DataLoader objects
train_loader = DataLoader(tokenized_data["train"], batch_size=16, shuffle=True)
test_loader = DataLoader(tokenized_data["test"], batch_size=16)


# %%
# Set up model
class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token's hidden state
        logits = self.classifier(pooled_output)
        return logits

# Initialize the model
model_name = "bert-base-uncased"
num_labels = 2  # Binary classification (positive/negative)
model = SentimentClassifier(model_name, num_labels)

#%%

print(model)# %%
# Set up training

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# %%
# Train the model

epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
