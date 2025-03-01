import os
import requests
from pathlib import Path
import concurrent.futures
from tqdm import tqdm  # Optional, for progress bar

# Base URL pattern for JPG files
base_url = "https://delta.hoddereducation.com/EInspection/9781036009007/files/ebook/files/assets/common/page-html5-substrates/page{page_num}_2.jpg?uni=79ebb642b5b1c39ea84ef2f79a2d9266"

# Create the book_jpg folder if it doesn't exist
book_folder = Path("book_jpg")
book_folder.mkdir(exist_ok=True)

# First page and last page numbers
first_page = 1
last_page = 554  # Based on the END LINK information

# Headers to mimic browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://delta.hoddereducation.com/EInspection/9781036009007/files/ebook/",
    "sec-ch-ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
    "sec-ch-ua-platform": '"macOS"',
}

# Authentication cookies
cookies = {
    "4cc9ba5d46fe4d208db22a9805776e8e": "[EIAuth/9781036009007/4cc9ba5d46fe4d208db22a9805776e8e]",
    "__cflb": "02DiuFSFPp8p1pCVBVUM7vmfkvwS2m8wSiV75JTBpdAXe",
}


# Function to verify if content is a JPG image
def is_jpg_content(content):
    # Check for JPEG magic numbers (FF D8 FF)
    return content.startswith(b"\xff\xd8\xff")


# Function to download a single page
def download_page(page_num):
    # Format the page number as a 4-digit string (e.g., 0001, 0002, ...)
    formatted_page = f"{page_num:04d}"
    url = base_url.format(page_num=formatted_page)

    # Create the output file path (using sequential numbering)
    output_file = book_folder / f"{page_num}.jpg"
    
    try:
        response = requests.get(url, headers=headers, cookies=cookies)

        # Check if the request was successful
        if response.status_code == 200:
            content = response.content

            # Verify if the content is a JPG image
            if is_jpg_content(content):
                # Save the JPG content to a file
                with open(output_file, "wb") as f:
                    f.write(content)
                return page_num, True, None
            else:
                # For debugging, save the non-JPG content to see what was received
                with open(book_folder / f"{page_num}_error.txt", "wb") as f:
                    f.write(content)
                return page_num, False, "Not a valid JPG image"
        else:
            # Save the error response for debugging
            with open(book_folder / f"{page_num}_error.html", "wb") as f:
                f.write(response.content)
            return page_num, False, f"Status code: {response.status_code}"

    except Exception as e:
        return page_num, False, str(e)


# Main execution with concurrent downloads
def main():
    # Create a thread pool with 50 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        # Submit all download jobs
        future_to_page = {
            executor.submit(download_page, page_num): page_num 
            for page_num in range(first_page, last_page + 1)
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_page), 
                          total=len(future_to_page), 
                          desc="Downloading pages"):
            page_num = future_to_page[future]
            try:
                page, success, error = future.result()
                if success:
                    print(f"Successfully downloaded page {page_num}")
                else:
                    print(f"Failed to download page {page_num}: {error}")
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")

    print("JPG download complete!")


if __name__ == "__main__":
    main()
