import torch
import torch.nn as nn  


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_classes):
        super(CNN_LSTM, self).__init__()
        self.input_size = input_size            
        self.output_size = output_size          
        self.hidden_size = hidden_size  

        self.cnn = nn.Conv1d(in_channels=input_size,
                             out_channels=output_size, 
                             kernel_size=3, 
                             stride=1, 
                             padding=0)
        
        self.lstm = nn.LSTM(input_size=output_size,
                            hidden_size=hidden_size, 
                            num_layers=1, 
                            dropout=0,
                            batch_first=True)
        
        self.Relu = nn.ReLU()

        self.fc = nn.Linear(in_features=hidden_size, 
                            out_features=num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x.shape = (B,seq_len,feature) = (64,30,99)
        feature 구성 : 21(joint 수)*4(좌표(x,y,z)+visibiliy) + 15(angle)
        """
        x = x.permute(0,2,1)        # (64,99,30)
        x = self.cnn(x)             # (64,64,28)
        x = self.Relu(x)            # (64,64,28)
        x = x.permute(0, 2, 1)      # (64,28,64)
        h_n, _ = self.lstm(x)       # (64,28,32)
        x = self.fc(h_n[:, -1, :])  
        x = self.softmax(x)         # (64,num_classes)
        return x