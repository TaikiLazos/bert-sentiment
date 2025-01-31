import wget
import polars as pl
import requests
import zipfile
import os
import logging
from bs4 import BeautifulSoup
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GDELT_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
OUTPUT_DIR = "gdelt_data_output"
INFER_SCHEMA_LENGTH = 10000

def fetch_latest_gdelt_fileURLs():
    """Get the latest GDELT 2.0 event and mention file URLs."""
    try:
        response = requests.get(GDELT_UPDATE_URL, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Error getting update file: {e}")

    event_file, mention_file = None, None
    
    for line in response.text.strip().split('\n'):
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[2].endswith('export.CSV.zip'):
            event_file = parts[2]
        elif parts[2].endswith('mentions.CSV.zip'):
            mention_file = parts[2]

    if not event_file or not mention_file:
        raise ValueError("Event or mention file URLs could not be found.")

    return event_file, mention_file

def download_and_extract_file(file_name):
    """Download and extract a GDELT file if it does not already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    zip_path = os.path.join(OUTPUT_DIR, os.path.basename(file_name))

    if os.path.exists(zip_path):
        logger.info(f"File already exists: {zip_path}, skipping download.")
    else:
        try:
            logger.info(f"Downloading {file_name}...")
            wget.download(file_name, zip_path)
            logger.info(f"\nDownloaded {zip_path}")
        except Exception as e:
            raise Exception(f"Failed to download {file_name}: {e}.")

    return extract_file(zip_path)

def extract_file(zip_path):
    """Extract a ZIP file only if the extracted file does not already exist."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        extracted_file = zip_ref.namelist()[0]
        extracted_path = os.path.join(OUTPUT_DIR, extracted_file)

        if os.path.exists(extracted_path):
            logger.info(f"Extracted file already exists: {extracted_path}, skipping extraction.")
        else:
            logger.info(f"Extracting {zip_path}...")
            zip_ref.extractall(OUTPUT_DIR)
            logger.info(f"Extracted to {OUTPUT_DIR}")

    return extracted_path

def load_gdelt_data(file_path, data_type):
    """Load GDELT data from a given file path into a DataFrame."""
    from gdelt_columns import columns_events, columns_mentions
    
    column_names = columns_events if data_type == "events" else columns_mentions
    
    df = pl.read_csv(
        file_path, 
        separator='\t', 
        infer_schema_length=INFER_SCHEMA_LENGTH,
        has_header=False,  
        new_columns=column_names  
    )
    
    return df

def scrape_article(url):
    """Scrape article title and text from URL using BeautifulSoup"""
    try:
        # Add delay to be nice to servers
        time.sleep(1)
        
        # Add headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Get the page
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get title (usually in title tag or h1)
        title = soup.title.string if soup.title else ''
        if not title:
            title = soup.find('h1').text if soup.find('h1') else ''
            
        # Get text (from paragraph tags)
        paragraphs = soup.find_all('p')
        text = ' '.join([p.text.strip() for p in paragraphs])
        
        return {
            'url': url,
            'title': title.strip(),
            'text': text.strip(),
            'success': True
        }
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return {
            'url': url,
            'title': '',
            'text': '',
            'success': False
        }

def process_and_save_data(events_df, mentions_df):
    """Process and save the filtered data to CSV with article content"""
    # Filter and join data
    filtered_events = events_df.select([
        'DATEADDED',
        'SOURCEURL',
        'ActionGeo_CountryCode',
        'GlobalEventID'
    ])
    
    filtered_mentions = mentions_df.select([
        'GlobalEventID',
        'MentionDocTone'
    ])
    
    # Join and remove duplicates
    merged_data = (filtered_events
                  .join(filtered_mentions, on='GlobalEventID', how='inner')
                  .unique(subset=['SOURCEURL']))  # Remove duplicates based on URL
    
    # Drop the ID column as we don't need it anymore
    final_data = merged_data.drop('GlobalEventID')
    
    # Get list of unique URLs
    urls = final_data['SOURCEURL'].unique().to_list()
    
    # Scrape articles
    logger.info(f"Scraping {len(urls)} articles...")
    articles = []
    for url in urls:
        result = scrape_article(url)
        articles.append(result)
        if result['success']:
            logger.info(f"Scraped: {result['title'][:50]}...")
        
    # Convert to DataFrame and merge with our data
    articles_df = pl.DataFrame(articles)
    final_data = final_data.join(
        articles_df,
        left_on='SOURCEURL',
        right_on='url',
        how='left'
    )
    
    # Save to CSV
    output_file = os.path.join(OUTPUT_DIR, "news_data_with_content.csv")
    final_data.write_csv(output_file)
    
    print(f"\nData saved to {output_file}")
    print(f"Total unique articles: {len(final_data)}")
    print(f"Successfully scraped: {sum(articles_df['success'])}")
    print("\nSample of saved data:")
    print(final_data.head())
    
    return final_data

def main():
    try:
        logger.info("Fetching latest GDELT files...")
        event_file, mention_file = fetch_latest_gdelt_fileURLs()
        
        logger.info("Getting event data ready...")
        event_path = download_and_extract_file(event_file)
        events_df = load_gdelt_data(event_path, "events")
        
        logger.info("Getting mention data ready...")
        mention_path = download_and_extract_file(mention_file)
        mentions_df = load_gdelt_data(mention_path, "mentions")
        
        # Process and save data
        process_and_save_data(events_df, mentions_df)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 