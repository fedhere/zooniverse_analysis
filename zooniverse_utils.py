                              
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from IPython.display import Image, display, HTML
import warnings
import json
import os
import time
from zooniverse_config import *



def createRBdf(zooniverse, multilabel, verbose=False, readsaved=False):
  
  #only use subjects with >1 label
  multilabelsubjects = zooniverse[multilabel].groupby(['subjects_id', 'label']).count()

  #get subjects to set as index
  indices = np.unique(multilabelsubjects.index.get_level_values('subjects_id').to_list())

  #create a df with index the subject and the number of R and B labels as columns
  rbdf = pd.DataFrame(index=indices, columns = ["Real_count", "Bogus_count"])

  print("assembling RB labels dataframe")
  if readsaved:
    rbdf = pd.read_csv("savedrbdf.csv", index_col=0)
  else:
    #if TEST: indices = indices[:100_000]
    for ind in tqdm(indices):      
      rbdf.loc[ind, 'Bogus_count'] = multilabelsubjects.loc[ind].loc['Bogus', 'classification_id']  \
        if 'Bogus' in multilabelsubjects.loc[ind].index else 0
      rbdf.loc[ind,'Real_count'] = multilabelsubjects.loc[ind].loc['Real', 'classification_id']  \
        if 'Real' in multilabelsubjects.loc[ind].index else 0
    #rbdf.to_csv("savedrbdf.csv")
  if verbose:
    print("\n\n\n#######################\nreal-bogus df head")
    display(rbdf.head())
  return rbdf, multilabelsubjects


def readinzooniverse(filein):
  print(f"reading in saved file {filein}")
  #zooniverse = pd.read_csv("zooniversedf_saved_02012026.csv", index_col=0)
  chunks = []
  
  for chunk in tqdm(pd.read_csv(filein, index_col=0, chunksize=chunk_size)):
      # Process each chunk as needed
      chunks.append(chunk)
    
  zooniverse = pd.concat(chunks).drop_duplicates()

  multilabel = zooniverse['countannotation'] > 1

  users = zooniverse["user_id"].unique()

  return zooniverse, multilabel, users


def load_posterior_dicts_dataframe(filename):
    """Load from Parquet and optionally convert back to dicts."""
    df = pd.read_parquet(filename)
    
    # Reset index for easier access
    df = df.reset_index()
    
    # Convert back to dictionaries if needed
    preals = {}
    skills = {}
    labels = {}
    subjects = sorted(df['subject'].unique())
    
    for subject in subjects:
        subject_data = df[df['subject'] == subject]
        preals[subject] = subject_data['preal'].dropna().tolist()
        skills[subject] = subject_data['skill'].dropna().tolist()
        labels[subject] = subject_data['label'].dropna().tolist()
    
    return {
        'df': df,
        'preals': preals,
        'skills': skills,
        'labels': labels,
        'subjects': subjects
    }



def load_user_histories_parquet(filename):
    """Load user histories from Parquet file."""

    
    df = pd.read_parquet(filename)

    """Convert DataFrame back to user_histories dictionary format."""
    user_histories = {}
    
    # Group by user_id
    for user_id, group in df.groupby('user_id'):
        # Sort by date to maintain chronological order
        group = group.sort_values('date')
        
        # Extract arrays
        user_histories[user_id] = {
            'dates': group['date'].values,
            'tp_rates': group['tp_rate'].values if 'tp_rate' in group.columns else \
          np.full(len(group), 0.5),
            'tn_rates': group['tn_rate'].values if 'tn_rate' in group.columns else \
          np.full(len(group), 0.5)
        }
        
        # Optional: Add counts if available
        for key in ['tp_counts', 'tn_counts', 'real_counts', 'bogus_counts']:
            if key in group.columns:
                user_histories[user_id][key] = group[key].values
    
    return user_histories


def save_parallel_dicts_to_dataframe(preals, skills, labels, subjects, filname):
    """Convert parallel dictionaries to a single DataFrame."""
    
    records = []
    
    for subject in subjects:
        # Get all lists for this subject
        preals_list = preals.get(subject, [])
        skills_list = skills.get(subject, [])
        labels_list = labels.get(subject, [])
        
        # Find max length
        max_len = max(len(preals_list), len(skills_list), 
                      len(labels_list))
        
        # Pad lists to same length with NaN
        for i in range(max_len):
            record = {
                'subject': subject,
                'step': i,
                'preal': preals_list[i] if i < len(preals_list) else np.nan,
                'skill': skills_list[i] if i < len(skills_list) else np.nan,
                'label': labels_list[i] if i < len(labels_list) else np.nan,
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    
    df.set_index(['subject', 'step'], inplace=True)
    
    df.to_parquet(filname)
    print(f"\n\n\n##### saved {len(subjects)} subjects with {len(df)} total rows")
    print(f"Average steps per subject: {len(df) / len(subjects):.1f}")
    
    return df



def save_user_histories_parquet(user_histories, filename):
    """Save user histories as Parquet file (columnar, compressed)."""
    records = []
    
    for user_id, history in user_histories.items():
        n = len(history['dates'])
        for i in range(n):
            record = {'user_id': user_id}
            
            # Handle dates - convert to timestamp if needed
            if isinstance(history['dates'][i], (np.datetime64, pd.Timestamp)):
                record['date'] = history['dates'][i]
            else:
                record['date'] = pd.to_datetime(history['dates'][i], unit='s')
            
            # Add available metrics
            for key in ['tp_rates', 'tn_rates', 'tp_counts', 'tn_counts', 
                       'real_counts', 'bogus_counts']:
                if key in history and i < len(history[key]):
                    record[key] = history[key][i]
            
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # Save as Parquet (compressed, preserves types)
    df.to_parquet(filename, index=False, compression='snappy')    
    print(f"Saved {len(df)} records to {filename} ({df.memory_usage().sum() / 1e6:.2f} MB)")

    return df



def show_image_from_url(url):
  import requests
  from PIL import Image
  from io import BytesIO
    
    
  response = requests.get(url)
  # Open with PIL
  img = Image.open(BytesIO(response.content))
  
  # Convert to numpy array for matplotlib
  img_array = np.array(img)

  # Display
  plt.figure()    
  plt.imshow(img_array)

def readoxfordlabels(filename="-atlas-eyeballers-rubin-difference-detectives-classifications.csv"):
  oxford = pd.read_csv(filename)
  
    
def get_url(df):
    return df['subject_data'].apply(lambda x: list(json.loads(x).values())[0]['URL'])

def show_triplet_and_posteriors(subject, zooniverse, posteriors=None,
                                posteriors_file="outputs/posteriorsdf_saved_02012026_short.pqt"):
  if posteriors is None:
    
    posteriors = load_posterior_dicts_dataframe(posteriors_file)
    
  zooniverse['URL'] = get_url(zooniverse)
  #       'df': df,
  #       'preals': preals,
  #       'skills': skills,
  #       'labels': labels,
  #       'subjects': subjects
  preals = posteriors['preals']
  skills = posteriors['skills']
  labels = posteriors['labels']    

  url = (zooniverse.loc[zooniverse["subject_ids"] == subject, "URL"].iloc[0])

  show_image_from_url(url)
  plt.figure()
  plt.plot(preals[subject], 'k-', label="posterior(R)")
  plt.plot(skills[subject], 'b.', label="IC skill", alpha=0.5)
  plt.plot(labels[subject], 'r.', label="label", alpha=0.5)
  plt.legend()
  plt.xlim(-0.1,17)
  plt.ylim(-0.05,1.05)
  plt.xlabel("classifications")
  plt.ylabel("posterior")
  plt.title(f"{subject}")
  plt.legend

  plt.show()
