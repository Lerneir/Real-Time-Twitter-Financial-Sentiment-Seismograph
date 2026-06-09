import requests
import base64

def push_to_github(csv_content: str, token: str, repo: str, branch: str, file_path: str) -> bool:
    """
    Pushes CSV content directly to a specified GitHub repository using the contents API.
    Handles existing files by fetching their SHA hash first.
    Returns True if successful, False otherwise.
    """
    if not token or not repo or repo == "username/repo":
        print("[GITHUB SYNC] GitHub credentials not fully configured. Skipping repository update.")
        return False
        
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # 1. Fetch current file SHA if it exists
    params = {"ref": branch} if branch else {}
    sha = None
    try:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception as e:
        print(f"[GITHUB SYNC] Error fetching SHA from GitHub API: {e}")
        return False
        
    # 2. Build upload payload
    try:
        content_b64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
        payload = {
            "message": "Update aggregated financial sentiment seismograph metrics",
            "content": content_b64,
            "branch": branch
        }
        if sha:
            payload["sha"] = sha
            
        put_r = requests.put(url, headers=headers, json=payload)
        if put_r.status_code in [200, 201]:
            print(f"[GITHUB SYNC] Successfully committed and pushed updated metrics to {repo}/{file_path}")
            return True
        else:
            print(f"[GITHUB SYNC] GitHub API Error ({put_r.status_code}): {put_r.text}")
            return False
    except Exception as e:
        print(f"[GITHUB SYNC] Exception occurred during file upload: {e}")
        return False
