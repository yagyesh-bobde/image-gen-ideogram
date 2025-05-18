import streamlit as st
import json
import requests
import time
import boto3
import io
import os
from PIL import Image
import base64
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Ideogram Image Generator with AR", layout="wide")

# Load environment variables
R2_PUBLIC_DOMAIN = os.getenv("R2_PUBLIC_DOMAIN")
AWS_REGION = os.getenv("AWS_REGION")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PROJECT_FOLDER = os.getenv("R2_PROJECT_FOLDER")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

# Function to call Ideogram Turbo API
def generate_image(prompt, api_token, is_first_or_last_page=False):
    api_url = "https://api.replicate.com/v1/models/ideogram-ai/ideogram-v2-turbo/predictions"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    
    # Set resolution based on page type
    # resolution = "1080x1440" if is_first_or_last_page else "1080x720"
    # width = 1080
    # height = 1440 if is_first_or_last_page else 720
    aspect_ratio = "3:4" if is_first_or_last_page else "3:2"
    
    payload = {
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "negative_prompt": "ugly, disfigured, low quality, blurry, nsfw"
        }
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 201 or response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error generating image: {response.text}")
        return None

# Function to check image generation status
def check_generation_status(prediction_id, api_token):
    api_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error checking status: {response.text}")
        return None

# Function to upload image to R2 storage
def upload_to_r2(image_url, r2_configs, book_title, page_number):
    try:
        # Download the image from the source URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return None
        
        # Create a file-like object from the image data
        image_data = io.BytesIO(response.content)
        
        # Configure S3 client to use R2
        s3_client = boto3.client(
            's3',
            endpoint_url=r2_configs['endpoint_url'],
            aws_access_key_id=r2_configs['access_key'],
            aws_secret_access_key=r2_configs['secret_key'],
            region_name=r2_configs['region']
        )
        
        # Sanitize book title for use in the filename
        sanitized_title = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in book_title)
        sanitized_title = sanitized_title.replace(' ', '_')
        
        # Define the key (path) for the image in the bucket
        key = f"ideogram/{sanitized_title}/page_{page_number}.jpg"
        
        # Upload the image to R2
        s3_client.upload_fileobj(
            image_data, 
            r2_configs['bucket_name'], 
            key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )
        
        # Construct and return the URL for the uploaded image
        image_url = f"{r2_configs['public_url']}/{key}"
        return image_url
    
    except Exception as e:
        st.error(f"Error uploading to R2: {e}")
        return None

def main():
    st.title("Ideogram Image Generator and R2 Uploader")
    
    # Use environment variables if available, otherwise show input fields
    api_key_default = REPLICATE_API_KEY if REPLICATE_API_KEY else ""
    r2_endpoint_default = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else ""
    r2_access_key_default = R2_ACCESS_KEY_ID if R2_ACCESS_KEY_ID else ""
    r2_secret_key_default = R2_SECRET_ACCESS_KEY if R2_SECRET_ACCESS_KEY else ""
    r2_region_default = AWS_REGION if AWS_REGION else "auto"
    r2_bucket_name_default = R2_BUCKET_NAME if R2_BUCKET_NAME else ""
    r2_public_url_default = R2_PUBLIC_DOMAIN if R2_PUBLIC_DOMAIN else ""
    
    # API credentials
    with st.sidebar:
        st.header("API Credentials")
        replicate_api_key = st.text_input("Replicate API Key", value=api_key_default, type="password")
        
        st.header("R2 Storage Credentials")
        r2_endpoint = st.text_input("R2 Endpoint URL", value=r2_endpoint_default)
        r2_access_key = st.text_input("R2 Access Key", value=r2_access_key_default, type="password")
        r2_secret_key = st.text_input("R2 Secret Key", value=r2_secret_key_default, type="password")
        r2_region = st.text_input("R2 Region", value=r2_region_default)
        r2_bucket_name = st.text_input("R2 Bucket Name", value=r2_bucket_name_default)
        r2_public_url = st.text_input("R2 Public URL", value=r2_public_url_default)
    
    # JSON Input
    st.header("Input JSON")
    json_input = st.text_area("Paste your JSON input here", height=200)
    uploaded_file = st.file_uploader("Or upload a JSON file", type=["json"])
    
    if uploaded_file is not None:
        json_input = uploaded_file.read().decode('utf-8')
    
    # Character customization
    st.header("Character Customization")
    character_name = st.text_input("Character Name", value="Mia")
    character_age = st.text_input("Character Age", value="2")
    
    # Process Button
    if st.button("Process JSON and Generate Images"):
        if not json_input:
            st.error("Please provide JSON input.")
            return
        
        if not replicate_api_key:
            st.error("Please provide Replicate API key.")
            return
            
        # Check R2 credentials
        r2_credentials_provided = all([r2_endpoint, r2_access_key, r2_secret_key, r2_bucket_name, r2_public_url])
        if not r2_credentials_provided:
            st.error("Please provide all R2 storage credentials.")
            return
        
        try:
            # Parse the JSON and replace placeholders
            data = json.loads(json_input)
            
            # Store R2 configurations
            r2_configs = {
                'endpoint_url': r2_endpoint,
                'access_key': r2_access_key,
                'secret_key': r2_secret_key,
                'region': r2_region,
                'bucket_name': r2_bucket_name,
                'public_url': r2_public_url
            }
            
            for book in data:
                # Only replace placeholders in image prompts
                for page in book['pages']:
                    page['imagePrompt'] = page['imagePrompt'].replace('{charactername}', character_name)
                    page['imagePrompt'] = page['imagePrompt'].replace('{characterage}', character_age)
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Track image generation status
            total_pages = sum(len(book['pages']) for book in data)
            completed = 0
            prediction_ids = {}
            
            # Initiate image generation for all pages
            status_text.text("Initiating image generation requests...")
            for book_idx, book in enumerate(data):
                book_title = book['title']
                total_pages_in_book = len(book['pages'])
                
                for page_idx, page in enumerate(book['pages']):
                    page_num = page['pageNumber']
                    prompt = page['imagePrompt']
                    
                    # Check if this is first or last page
                    is_first_or_last_page = page_idx == 0 or page_idx == total_pages_in_book - 1
                    
                    # Start image generation with appropriate resolution
                    prediction = generate_image(prompt, replicate_api_key, is_first_or_last_page)
                    
                    # With the "Prefer: wait" header, we might get the completed result directly
                    if prediction and 'output' in prediction:
                        if isinstance(prediction['output'], str):
                            prediction_ids[(book_idx, page_idx)] = {
                                'id': prediction.get('id', 'direct_completion'),
                                'status': 'succeeded',
                                'prompt': prompt,
                                'book_title': book_title,
                                'page_number': page_num,
                                'output_url': prediction['output']
                            }
                        elif isinstance(prediction['output'], list) and prediction['output']:
                            prediction_ids[(book_idx, page_idx)] = {
                                'id': prediction.get('id', 'direct_completion'),
                                'status': 'succeeded',
                                'prompt': prompt,
                                'book_title': book_title,
                                'page_number': page_num,
                                'output_url': prediction['output'][0]
                            }
                    # Otherwise, store the prediction ID for status checking
                    elif prediction and 'id' in prediction:
                        prediction_ids[(book_idx, page_idx)] = {
                            'id': prediction['id'],
                            'status': 'starting',
                            'prompt': prompt,
                            'book_title': book_title,
                            'page_number': page_num
                        }
                    
                    # Update progress
                    completed += 1
                    progress_bar.progress(completed / (total_pages * 2))  # First half for initiating
                    status_text.text(f"Initiated image generation: {completed}/{total_pages}")
                    
                    # Slight delay to avoid API rate limits
                    time.sleep(0.5)
            
            # Check status of all generations
            status_text.text("Checking image generation status...")
            all_completed = False
            
            while not all_completed:
                all_completed = True
                
                for (book_idx, page_idx), prediction_info in prediction_ids.items():
                    if prediction_info['status'] not in ['succeeded', 'failed']:
                        # Check the status
                        status = check_generation_status(prediction_info['id'], replicate_api_key)
                        
                        if status:
                            prediction_info['status'] = status['status']
                            
                            if status['status'] == 'succeeded':
                                # Store the output image URL from Replicate
                                # For the new API format, the output is a direct URL
                                if 'output' in status:
                                    if isinstance(status['output'], str):
                                        prediction_info['output_url'] = status['output']
                                    elif isinstance(status['output'], list) and status['output']:
                                        prediction_info['output_url'] = status['output'][0]
                                    else:
                                        prediction_info['output_url'] = None
                            elif status['status'] == 'failed':
                                st.error(f"Image generation failed for page {prediction_info['page_number']}")
                            else:
                                all_completed = False
                
                # Update status text
                completed_count = sum(1 for info in prediction_ids.values() if info['status'] == 'succeeded')
                status_text.text(f"Completed generations: {completed_count}/{total_pages}")
                
                # Update progress bar for second half
                progress_value = (completed + completed_count) / (total_pages * 2)
                progress_bar.progress(progress_value)
                
                if not all_completed:
                    time.sleep(3)  # Poll every 3 seconds
            
            # Upload completed images to R2 and update JSON
            status_text.text("Uploading images to R2 storage...")
            
            # Prepare to update the JSON with new image URLs
            updated_data = data.copy()
            
            # Use ThreadPoolExecutor for parallel uploads
            with ThreadPoolExecutor(max_workers=5) as executor:
                upload_futures = {}
                
                for (book_idx, page_idx), prediction_info in prediction_ids.items():
                    if prediction_info['status'] == 'succeeded' and 'output_url' in prediction_info:
                        # Submit the upload task to the executor
                        future = executor.submit(
                            upload_to_r2,
                            prediction_info['output_url'],
                            r2_configs,
                            prediction_info['book_title'],
                            prediction_info['page_number']
                        )
                        upload_futures[(book_idx, page_idx)] = future
                
                # Process the results as they complete
                for (book_idx, page_idx), future in upload_futures.items():
                    try:
                        r2_url = future.result()
                        if r2_url:
                            # Update the image URL in the JSON
                            updated_data[book_idx]['pages'][page_idx]['imageUrl'] = r2_url
                            
                            # Update progress
                            completed += 1
                            progress_value = completed / (total_pages * 2)
                            progress_bar.progress(progress_value)
                            status_text.text(f"Uploaded images: {completed - total_pages}/{total_pages}")
                    except Exception as e:
                        st.error(f"Error processing upload for book {book_idx}, page {page_idx}: {e}")
            
            # Complete the progress bar
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Display the updated JSON
            st.header("Results")
            st.subheader("Updated JSON with Image URLs")
            
            # Format the JSON nicely
            formatted_json = json.dumps(updated_data, indent=2)
            st.code(formatted_json, language="json")
            
            # Provide download link for the updated JSON
            st.download_button(
                label="Download Updated JSON",
                data=formatted_json,
                file_name="updated_storybook.json",
                mime="application/json"
            )
            
            # Display image previews
            st.subheader("Image Previews")
            cols = st.columns(3)
            
            image_count = 0
            for book in updated_data:
                for page in book['pages']:
                    if page['imageUrl'] != "TBD":
                        col_idx = image_count % 3
                        with cols[col_idx]:
                            st.image(
                                page['imageUrl'],
                                caption=f"Book: {book['title']}, Page: {page['pageNumber']}",
                                use_column_width=True
                            )
                            image_count += 1
            
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your input.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()