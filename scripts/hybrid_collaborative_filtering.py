import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define constants
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_PODCAST_DATA = '../data/podcast_df_040423.pkl'


class NNHybridFiltering(nn.Module):
    """
    Class for Hybrid Collaborative Filtering Neural Network model.
    Reference: AIPI540 Recommendation Systems Module notebook `nn_hybrid_recommender.ipynb`.
    """
    def __init__(self, n_users, n_podcasts, n_genres, n_producers, embdim_users, embdim_podcasts, embdim_genres, embdim_producers, n_activations, rating_range):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_embeddings=n_users,embedding_dim=embdim_users)
        self.podcast_embeddings = nn.Embedding(num_embeddings=n_podcasts,embedding_dim=embdim_podcasts)
        self.genre_embeddings = nn.Embedding(num_embeddings=n_genres,embedding_dim=embdim_genres)
        self.producer_embeddings = nn.Embedding(num_embeddings=n_producers,embedding_dim=embdim_producers)
        self.fc1 = nn.Linear(embdim_users+embdim_podcasts+embdim_genres+embdim_producers,n_activations)
        self.fc2 = nn.Linear(n_activations,1)
        self.rating_range = rating_range

    def forward(self, X):
        # Get embeddings for minibatch
        embedded_users = self.user_embeddings(X[:,0])
        embedded_podcasts = self.podcast_embeddings(X[:,1])
        embedded_genres = self.genre_embeddings(X[:,2])
        embedded_producers = self.producer_embeddings(X[:,3])
        # Concatenate user, podcast, genre, and producer embeddings
        embeddings = torch.cat([embedded_users,embedded_podcasts,embedded_genres,embedded_producers],dim=1)
        # Pass embeddings through network
        preds = self.fc1(embeddings)
        preds = F.relu(preds)
        preds = self.fc2(preds)
        # Scale predicted ratings to target-range [low,high]
        preds = torch.sigmoid(preds) * (self.rating_range[1]-self.rating_range[0]) + self.rating_range[0]
        return preds


def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=5, scheduler=None):
    """
    Trains a given neural network model with provided loss function, optimizer, dataloaders, device, and learning rate scheduler.
    Reference: AIPI540 Recommendation Systems Module notebook `nn_hybrid_recommender.ipynb`.

    Args:
        model(torch.nn.Module): Hybrid Collaborative Filtering Neural Network model
        criterion(torch.nn.*): Instance of loss function  
        optimizer(torch.optim): Instance of optimizer
        dataloaders(dict): Dictionary containing DataLoaders for both training and test data
        device(str): String indicating whether CPU or GPU is to be used for running the model
        num_epochs(int): Number of epochs to train the model for
        scheduler(torch.optim.lr_scheduler): Instance of learning rate scheduler

    Returns:
        model(torch.nn.Module): Hybrid Collaborative Filtering Neural Network model
        cost_paths(dict): Dictionary containing the training and validation losses
    """
    model = model.to(device) # Send model to GPU if available
    since = time.time()

    costpaths = {'train':[],'val':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Get the inputs and labels, and send to GPU if available
            for (inputs,labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs).view(-1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += np.sqrt(loss.item()) * labels.size(0)

            # Step along learning rate scheduler when in train
            if (phase == 'train') and (scheduler is not None):
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            costpaths[phase].append(epoch_loss)
            print('{} loss: {:.4f}'.format(phase, epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model, costpaths


def training_wrapper(df_podcast_enc):
    """
    Wrapper function for splitting data into train and test sets, putting them into dataloaders, and instantiating a Hybrid Collaborative Filtering neural network model for training.
    Reference: AIPI540 Recommendation Systems Module notebook `nn_hybrid_recommender.ipynb`.

    Args:
        df_podcast_enc(pd.DataFrame): Pandas dataframe containing the podcast dataset with encoded features

    Returns:
        model(torch.nn.Module): Hybrid Collaborative Filtering Neural Network model
        device(str): String indicating whether CPU or GPU is to be used for running the model
        cost_paths(dict): Dictionary containing the training and validation losses
    """
    # Specify features and target for training
    feats = ['user', 'itunes_id', 'genre', 'producer']
    target = 'rating'
    X = df_podcast_enc.loc[:, feats]
    y = df_podcast_enc.loc[:, target]
    # Split data into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=41, test_size=0.5)
    batchsize = 64
    trainloader, valloader = prep_dataloaders(X_train, y_train, X_val, y_val, batchsize)
    # Train the model
    dataloaders = {'train': trainloader, 'val': valloader}
    n_users = X.loc[:,'user'].max()+1
    n_podcast = X.loc[:,'itunes_id'].max()+1
    n_genres = X.loc[:,'genre'].max()+1
    n_producers = X.loc[:,'producer'].max()+1
    model = NNHybridFiltering(n_users,
                            n_podcast,
                            n_genres,
                            n_producers,
                            embdim_users=50,
                            embdim_podcasts=25, 
                            embdim_genres=5,
                            embdim_producers=25,
                            n_activations=50,
                            rating_range=[0.,5.])
    criterion = nn.MSELoss()
    lr=0.0001
    n_epochs=15
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cost_paths = train_model(model, criterion, optimizer, dataloaders, device, n_epochs, scheduler=None)
    return model, device, cost_paths


def prep_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    """
    Prepare PyTorch dataloaders from dataset for training and testing.
    
    Args:
        X_train(pd.DataFrame): Pandas dataframe containing the training features
        y_train(pd.DataFrame): Pandas dataframe containing the training target
        X_val(pd.DataFrame): Pandas dataframe containing the test features
        y_val(pd.DataFrame): Pandas dataframe containing the test target
        batch_size(int): Size of each mini-batch
    
    Returns:
        trainloader(torch.utils.data.DataLoader): DataLoader for training data
        valloader(torch.utils.data.DataLoader): DataLoader for test data
    """
    # Convert training and test data to TensorDatasets
    trainset = TensorDataset(torch.from_numpy(np.array(X_train)).long(), 
                            torch.from_numpy(np.array(y_train)).float())
    valset = TensorDataset(torch.from_numpy(np.array(X_val)).long(), 
                            torch.from_numpy(np.array(y_val)).float())

    # Create Dataloaders for our training and test data to allow us to iterate over minibatches 
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader


def encode_features(df):
    """
    Performs numerical/ordinal encoding on four features of the dataset, i.e. 'user', 'genre', 'itunes_id', and 'producer', and returns the encoded dataframe and encoders.
    
    Args:
        df(pd.DataFrame): Pandas dataframe containing the original podcast dataset
    
    Returns:
        df(pd.DataFrame): Pandas dataframe containing the podcast dataset with encoded features
        encoder_user(OrdinalEncoder): Encoder for user feature
        encoder_genre(OrdinalEncoder): Encoder for genre feature
        encoder_podcast(OrdinalEncoder): Encoder for itunes_id feature
        encoder_producer(OrdinalEncoder): Encoder for producer feature
    """
    # Encode user data
    encoder_user = OrdinalEncoder(dtype='int')
    encoder_user.fit(df[['user']])
    df[['user']] = encoder_user.transform(df[['user']])

    # Encode genre data
    encoder_genre = OrdinalEncoder(dtype='int')
    encoder_genre.fit(df[['genre']])
    df[['genre']] = encoder_genre.transform(df[['genre']])

    # Encode itunes_id data
    encoder_podcast = OrdinalEncoder(dtype='int')
    encoder_podcast.fit(df[['itunes_id']])
    df[['itunes_id']] = encoder_podcast.transform(df[['itunes_id']])

    # Encode producer data
    encoder_producer = OrdinalEncoder(dtype='int')
    encoder_producer.fit(df[['producer']])
    df[['producer']] = encoder_producer.transform(df[['producer']])

    return df, encoder_user, encoder_genre, encoder_podcast, encoder_producer


def plot_costs(costs):
    """
    Plots training and validation losses. Saves plot to a local image file.

    Args:
        costs(dict): Dictionary containing the training and validation losses
    
    Returns:
        None
    """
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    for i,key in enumerate(costs.keys()):
        ax_sub=ax[i%3]
        ax_sub.plot(costs[key])
        ax_sub.set_title(key)
        ax_sub.set_xlabel('Epoch')
        ax_sub.set_ylabel('Loss')
    plt.savefig('costs.png')


def predict_rating(model, user, itunes_id, genre, producer, user_encoder, genre_encoder, podcast_encoder, producer_encoder, device):
    """
    Predicts the rating of a podcast by a given user.
    Reference: AIPI540 Recommendation Systems Module notebook `nn_hybrid_recommender.ipynb`.

    Args:
        model(torch.nn.Module): Hybrid Collaborative Filtering Neural Network model
        user(str): Name of the user
        itunes_id(str): iTunes ID of the podcast
        genre(str): Genre of the podcast
        producer(str): Producer of the podcast
        user_encoder(OrdinalEncoder): Encoder for user feature
        genre_encoder(OrdinalEncoder): Encoder for genre feature
        podcast_encoder(OrdinalEncoder): Encoder for itunes_id feature
        producer_encoder(OrdinalEncoder): Encoder for producer feature
        device(str): String indicating whether CPU or GPU is to be used for running the model
    
    Returns:
        pred(np.ndarray): Numpy array containing the predicted rating of a podcast by the user
    """
    # Encode all data
    genre = genre_encoder.transform(np.array(genre).reshape(-1, 1))
    itunes_id = podcast_encoder.transform(np.array(itunes_id).reshape(-1, 1))
    user = user_encoder.transform(np.array(user).reshape(-1, 1))
    producer = producer_encoder.transform(np.array(producer).reshape(-1, 1))
    # Get predicted rating
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        X = torch.Tensor([user,itunes_id,genre,producer]).long().view(1,-1)
        X = X.to(device)
        pred = model.forward(X)
    return pred


def generate_recommendations(df, model, user, user_encoder, genre_encoder, podcast_encoder, producer_encoder, device, k=10):
    """
    Predicts the rating of a podcast by a given user.
    Reference: AIPI540 Recommendation Systems Module notebook `nn_hybrid_recommender.ipynb`.

    Args:
        df(pd.DataFrame): Pandas dataframe containing the podcast dataset
        model(torch.nn.Module): Hybrid Collaborative Filtering Neural Network model
        user(str): Name of the user
        user_encoder(OrdinalEncoder): Encoder for user feature
        genre_encoder(OrdinalEncoder): Encoder for genre feature
        podcast_encoder(OrdinalEncoder): Encoder for itunes_id feature
        producer_encoder(OrdinalEncoder): Encoder for producer feature
        device(str): String indicating whether CPU or GPU is to be used for running the model
        k(int): Number of recommendations to return, defaults to 10
    
    Returns:
        results(pd.DataFrame): Pandas Dataframe containing the recommended podcasts for the user
    """
    # Create progress bar
    pbar = tqdm(total=len(df))
    # Get predicted ratings for every podcast
    pred_ratings = []
    for podcast in df['itunes_id'].tolist():
        genre = df.loc[df['itunes_id']==podcast,'genre'].values[0]
        producer = df.loc[df['itunes_id']==podcast,'producer'].values[0]
        pred = predict_rating(model,user,podcast,genre,producer,user_encoder,genre_encoder,podcast_encoder,producer_encoder,device)
        pred_ratings.append(pred.detach().cpu().item())
        pbar.update()
    pbar.close()
    # Sort movies by predicted rating
    idxs = np.argsort(np.array(pred_ratings))[::-1]
    recs = df.iloc[idxs]['itunes_id'].values.tolist()
    # Filter out podcasts already listened to by user
    podcasts_listened = df.loc[df['user']==user, 'itunes_id'].tolist()
    recs = [rec for rec in recs if not rec in podcasts_listened]
    # Filter to top 10 recommendations
    recs = recs[:k]
    # Create a new dataframe to store recommendations
    results = pd.DataFrame()
    for rec in recs:
        results = pd.concat([results, df.loc[df['itunes_id']==rec,['title', 'producer', 'genre', 'description', 'link', 'itunes_id']]], axis=0)
    results = results.reset_index(drop=True)
    return results


def main(args):
    # Load pickled podcast dataframe
    df_podcast = pd.read_pickle(os.path.join(_CURRENT_DIR, _PODCAST_DATA))
    # Perform ordinal/numerical encoding of features
    df_podcast_enc, encoder_user, encoder_genre, encoder_podcast, encoder_producer = encode_features(df_podcast.copy())
    # Train the model
    model, device, costs = training_wrapper(df_podcast_enc)
    # Plot the cost over training and validation sets
    if args.plot_costs:
        plot_costs(costs)
    # Get recommendations for specified user
    df_podcast_uniq = df_podcast.drop_duplicates(subset=["itunes_id"],keep="first").reset_index(drop=True)
    results = generate_recommendations(df_podcast_uniq, model, args.user, encoder_user, encoder_genre, encoder_podcast, encoder_producer, device, args.k)
    # Display results as dataframe
    print(results)
    # Save results as CSV
    results.to_csv(args.file)
    print(f"Recommendations generated. Please check '{args.file}'.")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="Script to generate podcast recommendations for a given user based on hybrid collaborative filtering",
        epilog="Example usage: hybrid_collaborative_filtering.py --user 'ReddEye81' --k 10 --plot-costs"
    )
    parser.add_argument("--user", default="", help="Username of user that recommendations should be generated for. Defaults to empty string.")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations to return. Defaults to 10.")
    parser.add_argument("--file", default="hybrid_filtering_recommendations.csv", help="CSV file name to save recommendation results as. Defaults to 'hybrid_filtering_recommendations.csv'.")
    parser.add_argument("--plot-costs", default=False, action="store_true", help="Plot training and validation losses and save plot to an image file. Defaults to False.")
    args = parser.parse_args()
    print("Command Line Args: ", args)
    # Pass command line arguments to main function
    main(args)